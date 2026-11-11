"""Operator GUI: load a (preprocessed) trajectory, play it on the sim (or robot), watch it in 3D, record video.

Run from repo root:  .venv/bin/python -m src.xarm_pipeline.gui.app

Playback goes through the existing safety-checked scripts (play_cartesian_ik.sh / playback --control servo)
as a subprocess; the render follows the robot's measured joints on a separate read-only connection.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import json

import imageio.v2 as imageio
import numpy as np
from PyQt5 import QtCore, QtWidgets

from src.xarm_pipeline.gui.render import RobotScene
from src.xarm_pipeline.safety.config import load_safety_config

ROOT = Path(__file__).resolve().parents[3]
SIM_IP = "127.0.0.1"
ROBOT_IP_DEFAULT = os.environ.get("XARM_ROBOT_IP", "192.168.1.xxx")
POLL_MS = 33


def safety_config_for(path: Path) -> Path:
    """Frozen safety.yaml of the preprocessed set containing `path`, else the live config."""
    for parent in path.resolve().parents:
        if parent.parent.name == "preprocessed" and (parent / "safety.yaml").exists():
            return parent / "safety.yaml"
    return ROOT / "config" / "safety.yaml"


def list_parts(path: Path) -> list[str]:
    import pandas as pd
    cols = pd.read_csv(path, nrows=0).columns.str.strip()
    if "traj" not in cols:
        return []
    return list(dict.fromkeys(pd.read_csv(path, usecols=["traj"])["traj"].astype(str)))


def is_joint_file(path: Path) -> bool:
    import pandas as pd
    return "q1" in pd.read_csv(path, nrows=0).columns.str.strip()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("xArm 7 trajectory player")
        self.resize(1300, 850)
        self.traj_path: Path | None = None
        self.arm = None
        self.arm_ip = None
        self.writer = None
        self.proc = QtCore.QProcess(self)
        self.proc.setWorkingDirectory(str(ROOT))
        self.proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_output)
        self.proc.finished.connect(self._on_finished)

        cfg_path = ROOT / "config" / "safety.yaml"
        self.scene = RobotScene(self, load_safety_config(cfg_path))
        self._build_ui(cfg_path)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(POLL_MS)

    # ---- UI ---------------------------------------------------------------
    def _build_ui(self, cfg_path: Path) -> None:
        side = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(side)

        tgt = QtWidgets.QGroupBox("Target")
        tl = QtWidgets.QFormLayout(tgt)
        self.target = QtWidgets.QComboBox()
        self.target.addItems(["Simulator", "Real robot"])
        self.target.currentIndexChanged.connect(self._on_target_changed)
        self.robot_ip = QtWidgets.QLineEdit(ROBOT_IP_DEFAULT)
        self.robot_ip.setEnabled(False)
        self.conn_label = QtWidgets.QLabel("not connected")
        self.conn_label.setWordWrap(True)
        tl.addRow("Send to", self.target)
        tl.addRow("Robot IP", self.robot_ip)
        tl.addRow("Status", self.conn_label)
        lay.addWidget(tgt)

        trj = QtWidgets.QGroupBox("Trajectory")
        jl = QtWidgets.QVBoxLayout(trj)
        load = QtWidgets.QPushButton("Load CSV…")
        load.clicked.connect(self._load)
        self.file_label = QtWidgets.QLabel("(none)")
        self.file_label.setWordWrap(True)
        self.part = QtWidgets.QComboBox()
        self.cfg_label = QtWidgets.QLabel(f"safety: {cfg_path.relative_to(ROOT)}")
        self.cfg_label.setWordWrap(True)
        for w in (load, self.file_label, QtWidgets.QLabel("Part:"), self.part, self.cfg_label):
            jl.addWidget(w)
        lay.addWidget(trj)

        qbox = QtWidgets.QGroupBox("Queue (stops on first failure)")
        ql = QtWidgets.QVBoxLayout(qbox)
        self.queue = QtWidgets.QListWidget()
        self.queue.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.queue.setMinimumHeight(80)
        qb = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add selected")
        add.clicked.connect(self._queue_add)
        add_files = QtWidgets.QPushButton("Add files…")
        add_files.clicked.connect(self._queue_add_files)
        rm = QtWidgets.QPushButton("Remove")
        rm.clicked.connect(lambda: [self.queue.takeItem(self.queue.row(i)) for i in self.queue.selectedItems()])
        for b in (add, add_files, rm):
            qb.addWidget(b)
        self.run_queue_btn = QtWidgets.QPushButton("▶ Run queue")
        self.run_queue_btn.clicked.connect(self._run_queue)
        ql.addWidget(self.queue)
        ql.addLayout(qb)
        ql.addWidget(self.run_queue_btn)
        lay.addWidget(qbox)
        self._queue_pos: int | None = None
        self.tele_dir: Path | None = None
        self.tele_proc: QtCore.QProcess | None = None
        self.tele_runs: list[dict] = []

        run = QtWidgets.QGroupBox("Run")
        rl = QtWidgets.QVBoxLayout(run)
        self.play_btn = QtWidgets.QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._play)
        self.stop_btn = QtWidgets.QPushButton("■ Stop")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        self.ready_btn = QtWidgets.QPushButton("Go to ready pose")
        self.ready_btn.clicked.connect(lambda: self._go_ready())
        self.force_ready_btn = QtWidgets.QPushButton("Force ready pose (sim only, no wall check)")
        self.force_ready_btn.clicked.connect(lambda: self._go_ready(force=True))
        self.rec_auto = QtWidgets.QCheckBox("Record video while playing")
        self.telemetry_cb = QtWidgets.QCheckBox("Record telemetry (joints, TCP, commands) + comparison plots")
        for w in (self.play_btn, self.stop_btn, self.ready_btn, self.force_ready_btn, self.rec_auto, self.telemetry_cb):
            rl.addWidget(w)
        lay.addWidget(run)

        view = QtWidgets.QGroupBox("View")
        vl = QtWidgets.QVBoxLayout(view)
        self.walls_cb = QtWidgets.QCheckBox("Show safety walls")
        self.walls_cb.setChecked(True)
        self.walls_cb.toggled.connect(self.scene.set_walls_visible)
        self.rec_btn = QtWidgets.QPushButton("● Start recording")
        self.rec_btn.setCheckable(True)
        self.rec_btn.toggled.connect(self._toggle_record)
        reset = QtWidgets.QPushButton("Reset camera")
        reset.clicked.connect(lambda: (self.scene.plotter.view_isometric(), self.scene.plotter.render()))
        for w in (self.walls_cb, self.rec_btn, reset):
            vl.addWidget(w)
        lay.addWidget(view)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        lay.addWidget(self.log, 1)
        side.setMinimumWidth(320)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(340)

        self.scene.widget.setMinimumSize(300, 200)
        self.scene.widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self.scene.widget)
        split.addWidget(scroll)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setChildrenCollapsible(False)
        split.setSizes([900, 400])
        self.setCentralWidget(split)
        avail = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(min(1400, avail.width() - 40), min(900, avail.height() - 60))

    def _msg(self, text: str) -> None:
        self.log.appendPlainText(text.rstrip())

    # ---- connection / live joints ----------------------------------------
    def ip(self) -> str:
        return SIM_IP if self.target.currentIndex() == 0 else self.robot_ip.text().strip()

    def _on_target_changed(self, idx: int) -> None:
        if idx == 1:
            ok = QtWidgets.QMessageBox.warning(
                self, "Real robot", "Commands will be sent to the REAL robot.\n"
                "Check safety.yaml has measured values and the e-stop is in reach.",
                QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            if ok != QtWidgets.QMessageBox.Ok:
                self.target.setCurrentIndex(0)
                return
        self.robot_ip.setEnabled(idx == 1)
        self.force_ready_btn.setEnabled(idx == 0)
        self._disconnect()

    def _disconnect(self) -> None:
        if self.arm is not None:
            try:
                self.arm.disconnect()
            except Exception:
                pass
        self.arm, self.arm_ip = None, None
        self.conn_label.setText("not connected")

    def _ensure_arm(self) -> bool:
        ip = self.ip()
        if self.arm is not None and self.arm_ip == ip and self.arm.connected:
            return True
        self._disconnect()
        if getattr(self, "_next_try", 0) > time.time():
            return False
        self._next_try = time.time() + 3.0
        try:
            from xarm.wrapper import XArmAPI
            self.arm = XArmAPI(ip, baud_checkset=False, check_joint_limit=False)
            if not self.arm.connected:
                raise ConnectionError("SDK did not connect")
            self.arm_ip = ip
            self.conn_label.setText(f"connected {ip}")
            return True
        except Exception as e:  # noqa: BLE001
            self.arm = None
            self.conn_label.setText(f"no connection to {ip}: {e}")
            return False

    def _tick(self) -> None:
        if self._ensure_arm():
            code, q = self.arm.get_servo_angle(is_radian=True)
            if code == 0 and q is not None:
                self.scene.set_joints(np.asarray(q[:7], float))
                self.conn_label.setText(f"connected {self.arm_ip}  state={self.arm.state} err={self.arm.error_code}")
        if self.writer is not None:
            self.writer.append_data(self.scene.screenshot())

    # ---- trajectory -------------------------------------------------------
    def _load(self) -> None:
        start = ROOT / "data" / "preprocessed"
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load trajectory", str(start), "CSV (*.csv)")
        if not fn:
            return
        self.traj_path = Path(fn)
        self.file_label.setText(str(self.traj_path.relative_to(ROOT) if self.traj_path.is_relative_to(ROOT)
                                    else self.traj_path))
        self.part.clear()
        parts = list_parts(self.traj_path)
        self.part.addItem("(all parts in order)" if parts else "(single trajectory)")
        self.part.addItems(parts)
        cfg = safety_config_for(self.traj_path)
        self.cfg_label.setText(f"safety: {cfg.relative_to(ROOT)}")
        self.scene.set_config(load_safety_config(cfg))
        self.scene.set_walls_visible(self.walls_cb.isChecked())

    def _command(self, path: Path, part: str | None) -> tuple[str, list[str], dict]:
        spec = str(path) + (f"#{part}" if part else "")
        cfg = str(safety_config_for(path))
        env = {"XARM_IP": self.ip(), "XARM_SAFETY_CONFIG": cfg}
        if is_joint_file(path):
            py = str(ROOT / ".venv" / "bin" / "python")
            return py, ["-m", "src.xarm_pipeline.playback", "--trajectory", spec, "--control", "servo",
                        "--ip", self.ip(), "--safety-config", cfg], env
        return "bash", [str(ROOT / "scripts/control/play_cartesian_ik.sh"), spec, "servo"], env

    def _selected(self) -> tuple[Path, str | None]:
        return self.traj_path, (self.part.currentText() if self.part.currentIndex() > 0 else None)

    # ---- queue ------------------------------------------------------------
    def _queue_push(self, path: Path, part: str | None) -> None:
        label = (str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)) + (f"#{part}" if part else "")
        item = QtWidgets.QListWidgetItem(label)
        item.setData(QtCore.Qt.UserRole, (str(path), part))
        self.queue.addItem(item)

    def _queue_add(self) -> None:
        if self.traj_path is None:
            self._msg("load a trajectory first")
            return
        self._queue_push(*self._selected())

    def _queue_add_files(self) -> None:
        fns, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Add trajectories", str(ROOT / "data" / "preprocessed"),
                                                        "CSV (*.csv)")
        for fn in fns:
            self._queue_push(Path(fn), None)

    def _run_queue(self) -> None:
        if self.queue.count() == 0:
            self._msg("queue is empty")
            return
        if not self._confirm_robot():
            return
        self._queue_pos = 0
        self._telemetry_begin()
        self._play_queue_item()

    def _play_queue_item(self) -> None:
        for i in range(self.queue.count()):
            self.queue.item(i).setBackground(QtCore.Qt.transparent)
        item = self.queue.item(self._queue_pos)
        item.setBackground(QtCore.Qt.yellow)
        path, part = item.data(QtCore.Qt.UserRole)
        self._msg(f"== queue {self._queue_pos + 1}/{self.queue.count()}: {item.text()}")
        self._launch(Path(path), part)

    def _start_proc(self, prog: str, args: list[str], env: dict) -> None:
        pe = QtCore.QProcessEnvironment.systemEnvironment()
        pe.insert("PATH", f"{ROOT / '.venv' / 'bin'}:{pe.value('PATH')}")
        pe.insert("VIRTUAL_ENV", str(ROOT / ".venv"))
        for k, v in env.items():
            pe.insert(k, v)
        self.proc.setProcessEnvironment(pe)
        self._msg(f"$ {prog} {' '.join(args)}   [{self.ip()}]")
        self.proc.start(prog, args)
        self._set_running(True)

    def _confirm_robot(self) -> bool:
        return self.target.currentIndex() == 0 or QtWidgets.QMessageBox.question(
            self, "Confirm", f"Play on REAL robot at {self.ip()}?") == QtWidgets.QMessageBox.Yes

    def _play(self) -> None:
        if self.traj_path is None:
            self._msg("load a trajectory first")
            return
        if self._confirm_robot():
            self._telemetry_begin()
            self._launch(*self._selected())

    def _launch(self, path: Path, part: str | None) -> None:
        if self.rec_auto.isChecked() and not self.rec_btn.isChecked():
            self.rec_btn.setChecked(True)
        prog, args, env = self._command(path, part)
        if self.tele_dir is not None:
            env["XARM_TELEMETRY_DIR"] = str(self.tele_dir)
            self.tele_runs.append({"trajectory": str(path) + (f"#{part}" if part else ""),
                                   "safety_config": env["XARM_SAFETY_CONFIG"], "t_wall_start": time.time()})
            self._telemetry_write()
        self._start_proc(prog, args, env)

    def _go_ready(self, force: bool = False) -> None:
        args = [str(ROOT / "scripts/control/go_ready.sh"), "--yes"] + (["--force"] if force else [])
        env = {"XARM_IP": self.ip()}
        if self.traj_path is not None:
            env["XARM_SAFETY_CONFIG"] = str(safety_config_for(self.traj_path))
        self._start_proc("bash", args, env)

    def _stop(self) -> None:
        self._queue_pos = None
        if self.proc.state() != QtCore.QProcess.NotRunning:
            self.proc.terminate()  # scripts restore fence etc. on SIGTERM
            if not self.proc.waitForFinished(3000):
                self.proc.kill()
        if self.arm is not None:
            self.arm.set_state(4)
            self._msg("stop sent (state 4). Use 'Go to ready pose' to continue.")

    def _set_running(self, running: bool) -> None:
        self.play_btn.setEnabled(not running)
        self.run_queue_btn.setEnabled(not running)
        self.ready_btn.setEnabled(not running)
        self.force_ready_btn.setEnabled(not running and self.target.currentIndex() == 0)
        self.target.setEnabled(not running)
        self.stop_btn.setEnabled(running)

    def _on_output(self) -> None:
        self._msg(bytes(self.proc.readAllStandardOutput()).decode(errors="replace"))

    def _on_finished(self, code: int, _status) -> None:
        if self.tele_dir is not None and self.tele_runs and "exit_code" not in self.tele_runs[-1]:
            self.tele_runs[-1].update(t_wall_end=time.time(), exit_code=code)
            self._telemetry_write()
        self._msg(f"-- finished, exit code {code}" + {2: " (rejected by safety preprocessing)",
                                                      3: " (runtime safety stop)"}.get(code, ""))
        self._set_running(False)
        if self.rec_auto.isChecked() and self.rec_btn.isChecked():
            self.rec_btn.setChecked(False)
        if self._queue_pos is not None:
            item = self.queue.item(self._queue_pos)
            if item is not None:
                item.setBackground(QtCore.Qt.green if code == 0 else QtCore.Qt.red)
            self._queue_pos += 1
            if code != 0 or self._queue_pos >= self.queue.count():
                self._msg("== queue " + ("done" if code == 0 else "STOPPED on failure"))
                self._queue_pos = None
            else:
                QtCore.QTimer.singleShot(500, self._play_queue_item)
        if self._queue_pos is None:
            self._telemetry_end()

    # ---- telemetry --------------------------------------------------------
    def _telemetry_begin(self) -> None:
        if not self.telemetry_cb.isChecked() or self.tele_dir is not None:
            return
        self.tele_dir = ROOT / "outputs" / "telemetry" / time.strftime("%Y%m%d_%H%M%S")
        self.tele_dir.mkdir(parents=True, exist_ok=True)
        self.tele_runs = []
        self._telemetry_write()
        self.tele_proc = QtCore.QProcess(self)
        self.tele_proc.setWorkingDirectory(str(ROOT))
        self.tele_proc.start(str(ROOT / ".venv" / "bin" / "python"),
                             ["-m", "src.xarm_pipeline.telemetry", "record", "--ip", self.ip(), "--out", str(self.tele_dir)])
        self.tele_proc.waitForStarted(3000)
        time.sleep(0.5)  # let the sampler connect before the first command
        self._msg(f"telemetry -> {self.tele_dir}")

    def _telemetry_write(self) -> None:
        meta = {"ip": self.ip(), "target": self.target.currentText(), "runs": self.tele_runs}
        (self.tele_dir / "session.json").write_text(json.dumps(meta, indent=2))

    def _telemetry_end(self) -> None:
        if self.tele_dir is None:
            return
        if self.tele_proc is not None:
            self.tele_proc.terminate()
            self.tele_proc.waitForFinished(5000)
        d, self.tele_dir, self.tele_proc = self.tele_dir, None, None
        self._msg(f"telemetry saved: {d}; building comparison plots…")
        self._compare_proc = QtCore.QProcess(self)
        self._compare_proc.setWorkingDirectory(str(ROOT))
        self._compare_proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self._compare_proc.finished.connect(
            lambda c, _s, p=self._compare_proc: self._msg(bytes(p.readAll()).decode(errors="replace")
                                                         + f"-- comparison done (exit {c})"))
        self._compare_proc.start(str(ROOT / ".venv" / "bin" / "python"),
                                 ["-m", "src.xarm_pipeline.analysis.compare_run", str(d)])

    # ---- video ------------------------------------------------------------
    def _toggle_record(self, on: bool) -> None:
        if on:
            out = ROOT / "outputs" / "gui_videos"
            out.mkdir(parents=True, exist_ok=True)
            self.video_path = out / f"{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            self.writer = imageio.get_writer(self.video_path, fps=1000 // POLL_MS, macro_block_size=16)
            self.rec_btn.setText("■ Stop recording")
            self._msg(f"recording -> {self.video_path}")
        else:
            if self.writer is not None:
                self.writer.close()
                self._msg(f"saved {self.video_path}")
            self.writer = None
            self.rec_btn.setText("● Start recording")

    def closeEvent(self, ev) -> None:
        self.timer.stop()
        if self.rec_btn.isChecked():
            self.rec_btn.setChecked(False)
        if self.proc.state() != QtCore.QProcess.NotRunning:
            self._stop()
        if self.tele_proc is not None:
            self.tele_proc.terminate()
            self.tele_proc.waitForFinished(3000)
        self._disconnect()
        self.scene.close()
        super().closeEvent(ev)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

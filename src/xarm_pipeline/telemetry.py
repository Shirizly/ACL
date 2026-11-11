"""Run telemetry: robot feedback sampled on its own connection + per-command logs from the playback scripts.

A telemetry session directory (outputs/telemetry/<timestamp>/) holds:
  session.json           ip, target, and one entry per run: trajectory spec, wall start/end, exit code
  measured.csv           t_wall, q1..q7 [rad] (encoders), x,y,z [mm], roll,pitch,yaw [rad] (TCP), state, error
  commands_<k>.csv       t_wall, idx, t_target, x..yaw (Cartesian command, blank for joint runs), q1..q7 [rad]
                         (joint command; for Cartesian runs the joints the safety model expects the firmware IK to use)
  commands_<k>.json      trajectory spec, kind (cartesian|joint), control mode
  firmware_report.csv    controller real-time report (port 30000, ~250 Hz): t_wall, fw_t_us, state, mode, cmd_num,
                         tq1..tq7 / tdq1..tdq7 (firmware joint TARGET angle [rad] / velocity, i.e. what the controller
                         commands the joints, incl. its own IK in Cartesian modes), target TCP t_x..t_yaw, and actual
                         aq1..aq7, a_x..a_yaw, current1..7 (the simulator leaves the actual fields at zero / home pose)
All times are wall clock (time.time()), so the files line up without further bookkeeping.

The playback scripts write commands_<k> when XARM_TELEMETRY_DIR is set. Record feedback with:
  python -m src.xarm_pipeline.telemetry record --ip 127.0.0.1 --out <dir>      (stops on SIGTERM/SIGINT)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import time
from pathlib import Path

POSE_COLS = ("x", "y", "z", "roll", "pitch", "yaw")
Q_COLS = tuple(f"q{i}" for i in range(1, 8))
MEASURED_COLS = ("t_wall",) + Q_COLS + POSE_COLS + ("state", "error")
REPORT_COLS = (("t_wall", "fw_t_us", "state", "mode", "cmd_num") + tuple(f"t{c}" for c in Q_COLS)
               + tuple(f"td{c}" for c in Q_COLS) + tuple(f"t_{c}" for c in POSE_COLS)
               + tuple(f"a{c}" for c in Q_COLS) + tuple(f"a_{c}" for c in POSE_COLS)
               + tuple(f"current{i}" for i in range(1, 8)))
REPORT_PORT = 30000
COMMAND_COLS = ("t_wall", "idx", "t_target") + POSE_COLS + Q_COLS


class CommandLog:
    """Line-buffered CSV of sent commands; a no-op when XARM_TELEMETRY_DIR is unset."""

    def __init__(self, trajectory: str, kind: str, control: str) -> None:
        d = os.environ.get("XARM_TELEMETRY_DIR")
        self._fh = None
        if not d:
            return
        out = Path(d)
        out.mkdir(parents=True, exist_ok=True)
        k = 1
        while (out / f"commands_{k}.csv").exists():
            k += 1
        (out / f"commands_{k}.json").write_text(json.dumps(
            {"trajectory": trajectory, "kind": kind, "control": control, "t_wall_start": time.time()}, indent=2))
        self._fh = open(out / f"commands_{k}.csv", "w", newline="", buffering=1)
        self._w = csv.writer(self._fh)
        self._w.writerow(COMMAND_COLS)

    def log(self, idx: int, t_target: float, pose=None, q=None) -> None:
        if self._fh is None:
            return
        p = [f"{float(v):.6f}" for v in pose] if pose is not None else [""] * 6
        j = [f"{float(v):.6f}" for v in q[:7]] if q is not None else [""] * 7
        self._w.writerow([f"{time.time():.6f}", idx, f"{t_target:.6f}"] + p + j)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def record_report(ip: str, out: Path, stop: dict) -> None:
    """Log the controller's real-time report stream until stop['flag']."""
    import socket
    import struct

    from xarm.x3.report import ReportDataStructure

    try:
        sock = socket.create_connection((ip, REPORT_PORT), timeout=3)
    except OSError as e:
        print(f"telemetry: no real-time report on {ip}:{REPORT_PORT} ({e}); firmware targets not logged")
        return
    sock.settimeout(1.0)
    buf = b""

    def read(n: int) -> bytes:
        nonlocal buf
        while len(buf) < n:
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError("report socket closed")
            buf += chunk
        data, buf = buf[:n], buf[n:]
        return data

    rep = ReportDataStructure.create(REPORT_PORT)
    n = 0
    with open(out / "firmware_report.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(REPORT_COLS)
        while not stop["flag"]:
            try:
                head = read(4)
                body = read(struct.unpack(">I", head)[0] - 4)
            except (socket.timeout, ConnectionError):
                continue
            t = time.time()
            rep.update(head + body)
            vals = (list(rep.target_joint_angle) + list(rep.target_joint_velocity) + list(rep.target_tcp_pose)
                    + list(rep.actual_joint_angle) + list(rep.actual_tcp_pose) + list(rep.actual_joint_current))
            w.writerow([f"{t:.6f}", rep.timestamp, rep.state, rep.mode, rep.cmd_num] + [f"{float(v):.6f}" for v in vals])
            n += 1
            if n % 500 == 0:
                fh.flush()
    sock.close()
    print(f"telemetry: {n} firmware reports -> {out / 'firmware_report.csv'}")


def record(ip: str, out: Path, period_s: float = 0.004) -> None:
    from xarm.wrapper import XArmAPI

    out.mkdir(parents=True, exist_ok=True)
    arm = XArmAPI(ip, baud_checkset=False, check_joint_limit=False)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    import threading
    rt = threading.Thread(target=record_report, args=(ip, out, stop), daemon=True)
    rt.start()
    n = 0
    with open(out / "measured.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(MEASURED_COLS)
        while not stop["flag"]:
            t = time.time()
            cq, q = arm.get_servo_angle(is_radian=True)
            cp, p = arm.get_position(is_radian=True)
            if cq == 0 and cp == 0:
                w.writerow([f"{t:.6f}"] + [f"{v:.6f}" for v in q[:7]] + [f"{v:.6f}" for v in p[:6]]
                           + [arm.state, arm.error_code])
                n += 1
                if n % 200 == 0:
                    fh.flush()
            if period_s > 0:
                time.sleep(period_s)
    arm.disconnect()
    rt.join(timeout=3)
    print(f"telemetry: {n} samples -> {out / 'measured.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="sample joints + TCP pose until SIGTERM/SIGINT")
    r.add_argument("--ip", default=os.environ.get("XARM_IP", "127.0.0.1"))
    r.add_argument("--out", required=True)
    r.add_argument("--period", type=float, default=0.004, help="seconds between samples (SDK values refresh at ~250 Hz)")
    a = ap.parse_args()
    record(a.ip, Path(a.out), a.period)


if __name__ == "__main__":
    main()

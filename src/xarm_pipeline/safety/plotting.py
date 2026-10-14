"""Plots of a run's measured arm geometry vs the workcell walls (box frame).

Input: ``safety_monitor_log.csv`` written by the runtime monitor (measured
joints at ~250 Hz).  Output: a PNG with top view, two side views and the
clearance-vs-time curve, plus (optionally) an animated top/side view video.

    python -m src.xarm_pipeline.safety.plotting --run-dir outputs/safety_demo/<run>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import SafetyConfig, load_safety_config  # noqa: E402
from .geometry import Workcell  # noqa: E402

# Reference categorical palette (slots 1-3) + neutral inks.
C_TOOL = "#2a78d6"
C_ELBOW = "#eb6834"
C_WRIST = "#1baf7a"
C_STOP = "#e34948"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"
SURFACE = "#fcfcfb"

# chain point indices (geometry.POINT_NAMES)
P_SHOULDER, P_ELBOW, P_ELBOW_OFF, P_FORE, P_WRIST, P_WRIST_OFF, P_FLANGE, P_TIP = 1, 2, 3, 4, 5, 6, 7, 8
TRACES = ((P_TIP, "tool tip", C_TOOL), (P_ELBOW_OFF, "elbow", C_ELBOW), (P_FLANGE, "flange", C_WRIST))
VIEWS = (("Top view (x-y)", 0, 1), ("Side view (x-z)", 0, 2), ("Side view (y-z)", 1, 2))


def load_run(run_dir: Path):
    log = pd.read_csv(run_dir / "safety_monitor_log.csv")
    summ_path = run_dir / "safety_monitor_summary.json"
    summ = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    return log, summ


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)


def _draw_box(ax, wc: Workcell, cfg: SafetyConfig, a: int, b: int):
    lo, hi = wc.lo, wc.hi
    ax.add_patch(Rectangle((lo[a], lo[b]), hi[a] - lo[a], hi[b] - lo[b], fill=False, ec=INK, lw=1.6))
    m = cfg.safety_margin_mm
    ax.add_patch(Rectangle((lo[a] + m, lo[b] + m), hi[a] - lo[a] - 2 * m, hi[b] - lo[b] - 2 * m,
                           fill=False, ec=INK2, lw=0.9, ls="--"))
    pad = 120
    ax.set_xlim(lo[a] - pad, hi[a] + pad)
    ax.set_ylim(lo[b] - pad, hi[b] + pad)
    ax.set_aspect("equal")
    names = "xyz"
    ax.set_xlabel(f"world {names[a]} [mm]", color=INK2, fontsize=8)
    ax.set_ylabel(f"world {names[b]} [mm]", color=INK2, fontsize=8)


def _draw_arm(ax, P, a, b, color=INK, alpha=1.0, lw=2.0, radii=None):
    pts = P[1:]
    ax.plot(pts[:, a], pts[:, b], "-o", color=color, lw=lw, ms=3, alpha=alpha, zorder=4)


def plot_run(run_dir: str | Path, cfg: SafetyConfig | None = None, title: str | None = None) -> Path:
    run_dir = Path(run_dir)
    cfg = cfg or load_safety_config()
    wc = Workcell(cfg)
    log, summ = load_run(run_dir)
    Q = log[[f"q{i}" for i in range(1, 8)]].to_numpy()
    Pw = wc.points_world(Q)
    t = log["t"].to_numpy()
    cmin = log["min_clearance_mm"].to_numpy()
    i_min = int(np.argmin(cmin))
    trip = summ.get("trip") or None
    i_evt = i_min
    if trip and "t" in trip:
        i_evt = int(np.searchsorted(t, trip["t"]))
        i_evt = min(i_evt, len(t) - 1)

    fig = plt.figure(figsize=(15, 9.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1])
    for k, (name, a, b) in enumerate(VIEWS):
        ax = fig.add_subplot(gs[0, k])
        _style(ax)
        _draw_box(ax, wc, cfg, a, b)
        for idx, label, color in TRACES:
            ax.plot(Pw[:, idx, a], Pw[:, idx, b], color=color, lw=2, label=label, zorder=3)
        _draw_arm(ax, Pw[i_min], a, b, color=INK2, lw=1.5)
        if trip:
            _draw_arm(ax, Pw[i_evt], a, b, color=C_STOP, lw=2)
        # base footprint
        base = wc.base_to_world(np.zeros(3))
        ax.plot(base[a], base[b], "s", color=INK, ms=7, zorder=5)
        ax.set_title(name, color=INK, fontsize=10, loc="left")
        if k == 0:
            ax.legend(loc="upper left", fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[1, :])
    _style(ax)
    ax.plot(t, cmin, color=C_TOOL, lw=2, label="min clearance, any link (measured)")
    if "min_predicted_clearance_mm" in log:
        ax.plot(t, log["min_predicted_clearance_mm"], color=C_ELBOW, lw=1.2, alpha=0.8,
                label="predicted (lookahead)")
    ax.axhline(0, color=INK, lw=1.4)
    ax.text(t[-1], 2, "wall ", color=INK, fontsize=8, va="bottom", ha="right")
    ax.axhline(cfg.safety_margin_mm, color=INK2, lw=0.9, ls="--")
    ax.text(t[-1], cfg.safety_margin_mm + 2, f"safety margin {cfg.safety_margin_mm:.0f} mm ", color=INK2,
            fontsize=8, va="bottom", ha="right")
    ax.axhline(cfg.runtime.stop_margin_mm, color=INK2, lw=0.9, ls=":")
    ax.plot(t[i_min], cmin[i_min], "o", color=INK, ms=8, zorder=5)
    ax.annotate(f"min {cmin[i_min]:.1f} mm\n{log['pred_link'][i_min]} / {log['pred_wall'][i_min]}",
                (t[i_min], cmin[i_min]), xytext=(12, 18), textcoords="offset points", fontsize=8, color=INK)
    if trip:
        ax.axvline(trip["t"], color=C_STOP, lw=1.5)
        ax.text(trip["t"], ax.get_ylim()[1], f" STOP ({trip.get('source', '')})", color=C_STOP,
                fontsize=9, va="top")
    ax.set_xlabel("time since playback start [s]", color=INK2, fontsize=8)
    ax.set_ylabel("clearance to nearest wall [mm]", color=INK2, fontsize=8)
    ymin = min(-50.0, float(np.nanmin(cmin)) - 20)
    ax.set_ylim(ymin, min(max(200.0, float(np.nanmax(cmin)) + 20), 600))
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    head = title or run_dir.name
    sub = (f"measured min clearance {cmin[i_min]:.1f} mm ({'THROUGH the wall' if cmin[i_min] < 0 else 'inside'})"
           f"   |   {'STOPPED by runtime guard' if trip else 'no runtime stop'}"
           f"   |   grey arm = pose at min clearance" + ("; red arm = pose at stop" if trip else ""))
    fig.suptitle(f"{head}\n{sub}", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = run_dir / "safety_walls_plot.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return out


def animate_run(run_dir: str | Path, cfg: SafetyConfig | None = None, fps: int = 15,
                height_px: int = 900) -> Path:
    """Render a top+side view animation (mp4) of the measured arm vs walls."""
    import imageio_ffmpeg

    run_dir = Path(run_dir)
    cfg = cfg or load_safety_config()
    wc = Workcell(cfg)
    log, summ = load_run(run_dir)
    t = log["t"].to_numpy()
    Q = log[[f"q{i}" for i in range(1, 8)]].to_numpy()
    Pw = wc.points_world(Q)
    cmin = log["min_clearance_mm"].to_numpy()
    trip = summ.get("trip") or None
    frames_t = np.arange(0.0, t[-1] + 1e-9, 1.0 / fps)
    w_in, h_in = 8.0, 9.0
    dpi = height_px / h_in
    fig, axes = plt.subplots(2, 1, figsize=(w_in, h_in), facecolor=SURFACE)
    width = int(round(w_in * dpi)) // 2 * 2
    height = int(round(h_in * dpi)) // 2 * 2
    out = run_dir / "safety_walls_anim.mp4"
    writer = imageio_ffmpeg.write_frames(str(out), (width, height), fps=fps, codec="libx264",
                                         pix_fmt_out="yuv420p", macro_block_size=2)
    writer.send(None)
    try:
        for ft in frames_t:
            i = min(int(np.searchsorted(t, ft)), len(t) - 1)
            for ax, (name, a, b) in zip(axes, VIEWS[:2]):
                ax.clear()
                _style(ax)
                _draw_box(ax, wc, cfg, a, b)
                for idx, label, color in TRACES[:2]:
                    ax.plot(Pw[: i + 1, idx, a], Pw[: i + 1, idx, b], color=color, lw=1.5, alpha=0.7)
                stopped = trip is not None and ft >= trip["t"]
                _draw_arm(ax, Pw[i], a, b, color=C_STOP if stopped else INK, lw=2.5)
                ax.set_title(name, color=INK, fontsize=10, loc="left")
            state = "STOPPED by runtime guard" if (trip is not None and ft >= trip["t"]) else ""
            fig.suptitle(f"t={ft:5.2f}s  min clearance {cmin[i]:6.1f} mm  {state}", color=INK, fontsize=11)
            fig.set_size_inches(w_in, h_in)
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[:height, :width, :3]
            writer.send(np.ascontiguousarray(buf))
    finally:
        writer.close()
        plt.close(fig)
    return out


def compose_side_by_side(sim_video: Path, anim: Path, offset_s: float, out: Path,
                         crop: str = "500:900:1100:0") -> Path:
    """sim video (cropped to the Studio 3D/status panel of a 1600x900 recording)
    | animation, the animation delayed by ``offset_s`` seconds."""
    import subprocess

    import imageio_ffmpeg

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    pad = max(offset_s, 0.0)
    trim = max(-offset_s, 0.0)
    # NB: a trim filter in front of tpad (even trim=start=0) disables the padding
    # in ffmpeg 7, so only trim when the animation must start early.
    pre = f"trim=start={trim:.3f},setpts=PTS-STARTPTS," if trim > 0 else ""
    flt = (f"[0:v]crop={crop},scale=-2:900,setsar=1[a];"
           f"[1:v]{pre}tpad=start_duration={pad:.3f}:start_mode=clone:stop_duration=600:stop_mode=clone,"
           f"scale=-2:900,setsar=1[b];[a][b]hstack=inputs=2:shortest=1[v]")
    cmd = [ff, "-y", "-loglevel", "error", "-i", str(sim_video), "-i", str(anim), "-filter_complex", flt,
           "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", str(out)]
    subprocess.run(cmd, check=True)
    return out


def video_duration_s(path: Path) -> float:
    import re
    import subprocess

    import imageio_ffmpeg

    out = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out)
    if not m:
        raise RuntimeError(f"cannot read duration of {path}")
    h, mi, se = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(se)


def video_start_epoch(sim_video: Path) -> float:
    """Wall-clock time of the first frame: end time - duration (falls back to
    the page-creation time saved by record_web_sim)."""
    meta = json.loads(sim_video.with_suffix(".json").read_text())
    if "video_end_epoch" in meta:
        return meta["video_end_epoch"] - video_duration_s(sim_video)
    return meta["video_start_epoch"]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Plot measured arm geometry vs safety walls")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--title", default=None)
    p.add_argument("--animate", action="store_true")
    args = p.parse_args(argv)
    cfg = load_safety_config(args.config)
    print(f"saved plot: {plot_run(args.run_dir, cfg, args.title)}")
    if args.animate:
        print(f"saved animation: {animate_run(args.run_dir, cfg)}")


if __name__ == "__main__":
    main()

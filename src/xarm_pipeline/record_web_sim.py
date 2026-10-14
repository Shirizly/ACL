from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

from .sim_scene_overlay import arm_only_view, install_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record UFACTORY Studio web simulator to a video file"
    )
    parser.add_argument("--url", default="http://127.0.0.1:18333")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--output-dir", default="outputs/videos")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument(
        "--run-command",
        default=None,
        help="Optional shell command to run while recording",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=2.0,
        help="Delay after loading page before recording timeline starts",
    )
    parser.add_argument(
        "--orbit-drag",
        default="",
        help="Optional 'dx,dy' mouse drag (px) on the 3D view to rotate the camera, e.g. '120,60'",
    )
    parser.add_argument(
        "--zoom-steps",
        type=int,
        default=0,
        help="Optional number of mouse-wheel zoom-in steps on the 3D view",
    )
    parser.add_argument(
        "--stop-when-done",
        type=float,
        default=None,
        metavar="TAIL_SECONDS",
        help="Stop recording TAIL_SECONDS after --run-command exits (still capped by --seconds)",
    )
    parser.add_argument(
        "--arm-only",
        action="store_true",
        help="Fill the frame with the 3D arm view (covers the rest of the Studio UI)",
    )
    parser.add_argument(
        "--walls",
        action="store_true",
        help="Draw the safety workcell walls (+ visualization.objects) from the safety config into the 3D view",
    )
    parser.add_argument("--safety-config", default=None, help="Safety YAML for --walls/--arm-only (default config/safety.yaml)")
    parser.add_argument("--view-azimuth", type=float, default=35.0, help="--arm-only camera azimuth, deg from base +X toward +Y")
    parser.add_argument("--view-elevation", type=float, default=25.0, help="--arm-only camera elevation, deg")
    parser.add_argument("--view-zoom", type=float, default=None, help="--arm-only camera zoom (default: fit the workcell)")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Record with headless browser (no visible browser window)",
    )
    return parser.parse_args()


def _adjust_camera(page, orbit_drag: str, zoom_steps: int) -> None:
    """Rotate / zoom the Studio 3D view (orbit controls on its canvas)."""
    try:
        box = page.locator("canvas").first.bounding_box(timeout=3000)
    except Exception:
        box = None
    if not box:
        print("camera adjust skipped: 3D canvas not found")
        return
    cx = box["x"] + box["width"] / 2
    cy = box["y"] + box["height"] / 2
    if orbit_drag:
        dx, dy = (float(v) for v in orbit_drag.split(","))
        page.mouse.move(cx, cy)
        page.mouse.down()
        for i in range(20):
            page.mouse.move(cx + dx * (i + 1) / 20, cy + dy * (i + 1) / 20)
            time.sleep(0.02)
        page.mouse.up()
    page.mouse.move(cx, cy)
    for _ in range(max(zoom_steps, 0)):
        page.mouse.wheel(0, -300)
        time.sleep(0.1)
    time.sleep(0.5)


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=args.headless,
            args=[f"--window-size={args.width},{args.height}"],
        )
        context = browser.new_context(
            viewport={"width": args.width, "height": args.height},
            record_video_dir=str(out_dir),
            record_video_size={"width": args.width, "height": args.height},
        )
        page = context.new_page()
        # Wall-clock time when the page was created. Playwright's first frame
        # comes ~2-3 s later (first paint), so to align other logs with the
        # video prefer video_end_epoch - video duration.
        video_start_epoch = time.time()
        page.goto(args.url, wait_until="domcontentloaded")

        # Dismiss the known simulator popup if present so the robot view is visible.
        try:
            page.get_by_role("button", name="Close").first.click(timeout=2000)
        except Exception:
            pass

        # Give Studio time to initialize before timing and optional playback.
        time.sleep(args.warmup_seconds)

        if args.walls:
            print(f"walls overlay: {install_overlay(page, args.safety_config)}")
        if args.arm_only:
            print(
                "arm-only view: "
                + arm_only_view(
                    page,
                    azimuth_deg=args.view_azimuth,
                    elevation_deg=args.view_elevation,
                    zoom=args.view_zoom,
                    config_path=args.safety_config,
                )
            )
        elif args.orbit_drag or args.zoom_steps:
            _adjust_camera(page, args.orbit_drag, args.zoom_steps)

        process: subprocess.Popen[str] | None = None
        run_log_path: Path | None = None
        run_log_file = None
        if args.run_command:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_log_path = out_dir / f"{timestamp}_run_command.log"
            run_log_file = run_log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                args.run_command,
                shell=True,
                stdout=run_log_file,
                stderr=subprocess.STDOUT,
            )

        t0 = time.perf_counter()
        done_at = None
        while (time.perf_counter() - t0) < args.seconds:
            time.sleep(0.05)
            if process is not None and args.stop_when_done is not None and process.poll() is not None:
                done_at = done_at or time.perf_counter()
                if time.perf_counter() - done_at >= args.stop_when_done:
                    break

        timed_out = bool(process and process.poll() is None)
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

        if run_log_file is not None:
            run_log_file.flush()
            run_log_file.close()

        page_title = page.title() if page else "studio"
        video_end_epoch = time.time()  # the last video frame is (about) now
        context.close()
        video_path = page.video.path()
        browser.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_name = f"{timestamp}_{page_title.lower().replace(' ', '_')}.webm"
    final_path = out_dir / final_name
    Path(video_path).rename(final_path)
    final_path.with_suffix(".json").write_text(
        json.dumps({"video_start_epoch": video_start_epoch, "video_end_epoch": video_end_epoch}),
        encoding="utf-8",
    )
    print(f"Saved recording: {final_path}")
    if run_log_path is not None:
        print(f"Saved run log: {run_log_path}")
    if process is not None:
        # Pass the run command's outcome on (124 = still running when the recording time ran out).
        rc = 124 if timed_out else int(process.returncode or 0)
        if rc != 0:
            print(f"run command exited with code {rc}" + (" (stopped: recording time ran out)" if timed_out else ""))
        sys.exit(rc)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright


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
        "--headless",
        action="store_true",
        help="Record with headless browser (no visible browser window)",
    )
    return parser.parse_args()


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
        page.goto(args.url, wait_until="domcontentloaded")

        # Dismiss the known simulator popup if present so the robot view is visible.
        try:
            page.get_by_role("button", name="Close").first.click(timeout=2000)
        except Exception:
            pass

        # Give Studio time to initialize before timing and optional playback.
        time.sleep(args.warmup_seconds)

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
        while (time.perf_counter() - t0) < args.seconds:
            time.sleep(0.05)

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
        context.close()
        video_path = page.video.path()
        browser.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_name = f"{timestamp}_{page_title.lower().replace(' ', '_')}.webm"
    final_path = out_dir / final_name
    Path(video_path).rename(final_path)
    print(f"Saved recording: {final_path}")
    if run_log_path is not None:
        print(f"Saved run log: {run_log_path}")


if __name__ == "__main__":
    main()

# xArm 7 Simulation + Control Pipeline

This workspace sets up a development pipeline to run UFACTORY Studio simulation in Docker and execute joint and Cartesian trajectory playback from your host or dev container.

## Why this approach

- Uses the official shared simulator image from your guide.
- Keeps simulator lifecycle in one Compose file.
- Keeps control scripts in versioned Python files.
- Works for simulator now and real robot later by changing IP.

## Prerequisites

- Docker Engine with Compose plugin (`docker compose`).
- VS Code with **Dev Containers** extension (recommended).
- For video capture: Playwright Chromium browser (`python -m playwright install chromium`).

## Quick start

1. Open this folder in VS Code.
2. Reopen in container: Command Palette -> `Dev Containers: Reopen in Container`.
3. Start simulator container:

   ```bash
   scripts/sim/up.sh
   ```

4. Start xArm 7 firmware + studio inside simulator container:

   ```bash
   scripts/sim/start_xarm7.sh
   ```

5. Open web UI at `http://127.0.0.1:18333`.
6. In another terminal, generate a sample trajectory:

   ```bash
   python -m src.xarm_pipeline.generate_sample_trajectory
   ```

7. Run position control playback:

   ```bash
   scripts/control/play_position.sh
   ```

8. Run velocity control playback (requires `dq1..dq7` columns):

   ```bash
   scripts/control/play_velocity.sh
   ```

9. Generate a sample Cartesian trajectory:

   ```bash
   scripts/control/gen_cartesian_sample.sh
   ```

10. Run Cartesian trajectory using firmware IK (`servo` mode by default):

   ```bash
   scripts/control/play_cartesian_ik.sh
   ```

11. Optional Cartesian modes:

   ```bash
   scripts/control/play_cartesian_ik.sh data/trajectories/sample_cartesian_circle.csv pose
   scripts/control/play_cartesian_ik.sh data/trajectories/sample_cartesian_circle.csv velocity
   ```

12. Cartesian variant with optional gripper orientation input:

   ```bash
   scripts/control/play_cartesian_with_gripper_orientation.sh data/trajectories/your_traj.csv servo
   ```

   If trajectory omits orientation columns, provide fallback orientation (radians):

   ```bash
   scripts/control/play_cartesian_with_gripper_orientation.sh data/trajectories/xyz_only.csv servo "3.14159,0,0"
   ```

## Render and record simulation video

1. Ensure simulator and firmware are running:

   ```bash
   scripts/sim/up.sh
   scripts/sim/start_xarm7.sh
   ```

2. Install Playwright browser once:

   ```bash
   python -m playwright install chromium
   ```

3. Record current web simulation for 20 seconds:

   ```bash
   scripts/sim/record_web_video.sh 20
   ```

4. Run Cartesian IK playback and record in one command:

   ```bash
   scripts/control/play_cartesian_ik_record.sh data/trajectories/sample_cartesian_circle.csv servo 30
   ```

Videos from analysis runs are saved inside each report directory as `.webm` files.

## Commanded vs measured trajectory error report

For accurate error analysis, this project compares commanded Cartesian points against sampled robot feedback from the SDK. Video alone is not used for numeric error.

Run a trajectory with recording and automatic report generation:

```bash
scripts/control/play_cartesian_ik_record.sh data/trajectories/sample_cartesian_circle.csv servo 30
```

That wrapper now does two phases:

- an off-record prestart move to the trajectory's first point, with a short settle delay
- the recorded motion, which begins with a 0.5 s hold at that initial point before motion starts

This now saves a run directory under `outputs/reports/<timestamp>_<mode>/` with:

- `commanded_cartesian.csv`: Cartesian commands with target and send timestamps.
- `measured_cartesian.csv`: high-rate sampled robot feedback (`--feedback-sample-period 0.0` means max possible polling rate).
- `cartesian_error_timeseries.csv`: aligned error signals.
- `cartesian_tracking_error.png`: figure with absolute error norm and per-axis errors (`x,y,z`).
- `cartesian_error_metrics.json`: summary metrics.
- `*_run_command.log`: execution log.
- `*_ufactory_studio.webm`: simulator video for visual confirmation.

Safety defaults now enabled for Cartesian playback:

- Orientation continuity with per-step clamp (`--max-orientation-step-deg`, default `8`).
- Joint soft-limit guard (`--joint-limit-deg`, default `365`, mode `--joint-limit-mode raw`) checked every command.

If a command sequence risks sending joints near +/-360 deg, playback aborts early with a clear error in the run log.

You can re-run analysis manually:

```bash
python -m src.xarm_pipeline.analyze_cartesian_error \
   --commanded outputs/reports/<run>/commanded_cartesian.csv \
   --measured outputs/reports/<run>/measured_cartesian.csv \
   --out-dir outputs/reports/<run>
```

## Real robot later

Set robot IP and reuse the same scripts:

```bash
export XARM_IP=192.168.1.XXX
scripts/control/play_position.sh data/trajectories/your_recorded.csv
```

## File map

- `docker-compose.sim.yml`: Simulator container and ports.
- `.devcontainer/`: Reproducible dev environment with Python and Docker CLI.
- `scripts/sim/`: Simulator lifecycle helpers.
- `scripts/control/`: Entry-point scripts for playback.
- `src/xarm_pipeline/`: Python trajectory I/O and playback logic.
- `docs/trajectory_format.md`: Required CSV format.

## Firmware IK note

For Cartesian trajectories, this project intentionally sends Cartesian commands (`set_position`/`set_servo_cartesian`) to xArm firmware. IK is solved by the robot firmware in the simulator container, which mirrors how execution works on the real arm.

## Notes from UFACTORY guide

- Simulator IP from host is `127.0.0.1`.
- For external IDE execution, keep `check_joint_limit=False`.
- If Studio shows SN warning, close it and continue.

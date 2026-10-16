---
name: xarm-pipeline
description: Orientation for the ACL xArm 7 repo — UFACTORY simulator in Docker, joint/Cartesian CSV trajectory playback via xarm-python-sdk, video recording, error reports, safety walls. Use at the start of any task in this repo (running the sim, playing/recording trajectories, editing playback code, trajectory formats, safety).
---

# xArm 7 sim + playback pipeline (ACL repo)

Python pipeline that replays CSV trajectories on a UFACTORY **xArm 7** — the
official simulator now, the real arm later (same code, different IP).

## Layout

| Path | What |
|---|---|
| `src/xarm_pipeline/trajectory_io.py` | CSV loaders → `JointTrajectory` / `CartesianTrajectory` dataclasses |
| `src/xarm_pipeline/playback.py` | Joint playback: `--control position` (mode 0, `set_servo_angle`) / `velocity` (mode 4, `vc_set_joint_velocity`) |
| `src/xarm_pipeline/cartesian_playback.py` | Cartesian playback, firmware IK: `servo` (mode 1, `set_servo_cartesian`, default), `pose` (mode 0, `set_position`), `velocity` (mode 5, `vc_set_cartesian_velocity`); feedback sampler thread, report dir, orientation unwrap/clamp, joint soft-limit check |
| `src/xarm_pipeline/analyze_cartesian_error.py` | Commanded vs measured TCP error → CSV/PNG/JSON |
| `src/xarm_pipeline/record_web_sim.py` | Playwright records the Studio web UI (`.webm`), optionally running a command meanwhile |
| `src/xarm_pipeline/validate_trajectory.py` | Format pre-flight check for CSVs (no robot needed) |
| `src/xarm_pipeline/generate_sample_*.py`, `scripts/control/gen_*.sh` | Sample trajectory generators |
| `scripts/sim/` | `up.sh`, `down.sh`, `start_xarm7.sh`, `status.sh`, `shell.sh`, `record_web_video.sh` |
| `scripts/control/` | `play_position.sh`, `play_velocity.sh`, `play_cartesian_ik.sh`, `play_cartesian_ik_record.sh` (prestart move + video + report), `prepare_cartesian_start.sh` |
| `docs/trajectory_format.md` | **Authoritative trajectory CSV spec** (columns, units, frames, timing semantics, exporter checklist) |
| `data/trajectories/` | Sample CSVs (`sample_sine_joint.csv`, `sample_cartesian_circle.csv`, `cartesian_bigmove.csv`) |
| `outputs/` | Git-ignored. `reports/<ts>_<mode>/` run reports + videos, `videos/` |

## Environment

- Use `.venv/bin/python` (py3.13; numpy, pandas, xarm-python-sdk 1.18.4, playwright+chromium, matplotlib, pyyaml). Run modules from repo root: `.venv/bin/python -m src.xarm_pipeline.<module>`. Shell scripts call plain `python`, so activate the venv first (`. .venv/bin/activate`).
- Robot IP via `XARM_IP` env / `--ip` (sim: `127.0.0.1`).
- Nothing committed yet (fresh repo, branch `main`).

## Simulator

- Image `danielwang123321/uf-ubuntu-docker`, container `uf_software`, Studio web UI http://127.0.0.1:18333, SDK ports 502–504, 30000–30003.
- Start: `scripts/sim/up.sh`, then firmware. Check first with `docker exec uf_software ps -e | grep controller`: the container survives reboots with firmware running. If the SDK can't connect (`failed to check version`), `docker restart uf_software` and start the firmware once; the arm is then at the zero pose, where the 172 mm tool model is below the floor, so do a plain joint move to the ready pose (0,-28.17,0,23.24,0,51.41,0)° before safety-checked runs. `scripts/sim/start_xarm7.sh` uses `docker exec -it`, which **fails from a non-interactive shell** — from Claude use `docker exec -d uf_software /xarm_scripts/xarm_start.sh 7 7` and poll until `XArmAPI('127.0.0.1')` connects.
- Recording from Claude: pass `--headless` to `record_web_sim`. `--arm-only --walls` gives a full-frame 3D view with the safety box + `visualization.objects` drawn in (`src/xarm_pipeline/sim_scene_overlay.py`, injected into Studio's three.js scene; visual only). If Studio shows an SN warning popup, it is dismissed automatically (or ignore it).
- After an e-stop / error in the sim: `arm.clean_error(); arm.clean_warn(); arm.motion_enable(True); arm.set_mode(0); arm.set_state(0)`.
- Always construct `XArmAPI(ip, baud_checkset=False, check_joint_limit=False)` (per UFACTORY guide for external control).

## Conventions (details in docs/trajectory_format.md)

- Robot base frame; positions **mm**, angles **rad** (SDK called with `is_radian=True`).
- Grouped files: optional `traj` column holds several named trajectories; select anywhere a path is accepted with `file.csv#name` or `file.csv#<0-based index>` (`trajectory_io.read_trajectory_frame`). Columns blank for a whole trajectory count as absent. List contents: `python -m src.xarm_pipeline.trajectory_io file.csv`. Sample: `data/trajectories/sample_bank_cartesian.csv`.
- Cartesian orientation = xArm RPY: fixed-axis X→Y→Z, `R = Rz(yaw)·Ry(pitch)·Rx(roll)` (scipy `from_euler('xyz', ...)`, lowercase) — verified against sim firmware FK. Pose refers to the TCP = flange + firmware TCP offset (SDK 1.18.4 has no `get_tcp_offset`; read it in Studio).
- All-zeros joints = folded pose, flange at (206, 0, 120.5) mm pointing down (rpy π,0,0). Shoulder (J2) 267 mm above base origin, reach ≈700 mm.
- Firmware FK/IK without moving: `arm.get_forward_kinematics(q, input_is_radian=True, return_is_radian=True)`, `arm.get_inverse_kinematics(pose, ...)`. Sim firmware reports v2.4.0, device_type 7.
- `q1..q7` base→wrist. J4 range differs by hardware revision (SDK `x_config.py` JOINT_LIMITS) — check the real arm's type.

## Gotchas

- `playback.py --control position` uses mode 0 + `set_servo_angle(wait=False)`: commands are **queued planned moves at the SDK default 20°/s**, so CSV timing is not honored. True timed joint streaming would need mode 1 + `set_servo_angle_j`.
- `cartesian_playback.py` silently rewrites orientation: unwraps to be near the previous sample (starting from the robot's current pose) and **clamps each step to `--max-orientation-step-deg` (8°)**.
- Cartesian timeline is re-based: sample 0 at t=0, then a `--start-hold-seconds` (0.5 s) hold, then sample i at `hold + (t[i] - t[1])`.
- The loaders do not reject NaN in x/y/z or q, non-monotonic `t`, or whitespace-padded headers; run `validate_trajectory` first.
- `pose` mode (`set_position`, wait=False) also queues linear moves at default 100 mm/s — use `servo` for timed playback.

## Safety walls (docs/safety_walls.md is the full write-up)

- Config: `config/safety.yaml` (mm/deg/s). Workcell box in a floor frame (origin = box floor center, D along x, W along y, H up) + robot base pose (x,y,z,yaw) in it, margin, tool capsule, link radii, monitor timing. Values tagged `[MEASURE]` are **sim placeholders** (box 1200×800×1100, base at origin, 172 mm gripper) until the user supplies real measurements.
- Code: `src/xarm_pipeline/safety/` — `kinematics.py` (local xArm 7 FK/IK, DH verified vs firmware to 2e-4 mm), `geometry.py` (8 capsules vs box), `preprocess.py` (densified check; policy `reject`|`clip`), `runtime.py` (per-command check + monitor thread on its own connection, stops with `set_state(4)` then `emergency_stop()`), `firmware_fence.py` (xArm TCP-only boundary, set at playback start, restored on exit), `session.py`, `plotting.py`, `demo.py`.
- On by default in `playback.py` and `cartesian_playback.py`. Independent opt-outs (loud banners): `--disable-safety-preprocess`, `--disable-runtime-guard`, `--disable-firmware-fence`; `--safety-policy clip`. Exit codes: 2 = rejected by preprocessing, 3 = safety stop at runtime.
- Offline check: `scripts/control/validate_trajectory_safety.sh <csv> [reject|clip]` (set `XARM_IP` to seed Cartesian IK from the robot's real joints — results depend on start state).
- Fitting: `safety/fit.py` re-places Cartesian trajectories by a rigid motion (translation + yaw; optional tool spin/tilt) to maximize full-arm wall slack near `fit.preferred_center_base_mm`, splits ones that can't fit (`name__pK`), and stores the validated start joints. ~15-45 s/trajectory; minutes when splitting.
- Preprocessed sets (`safety/preprocessed.py`): `data/preprocessed/NNN_D..xW..xH../` per setup fingerprint (workcell, base, margin, geometry, preprocess tolerances, `fit:`). Each has a frozen `safety.yaml`, `meta.json`, `fit_log.jsonl`, and clean copies at mirrored paths (`external/` for files outside data/). **Cartesian playback runs the clean copy by default** (fits on first use, refits if the source SHA-1 changed); `--no-fit` plays raw, `--refit` forces. Clean copies start via two joint moves (ready pose -> stored start joints, both safety-checked) in `prepare_cartesian_start.sh`, because the firmware's linear approach leaves the redundant elbow elsewhere. `python -m src.xarm_pipeline.safety.preprocessed list|which|resolve`. `data/preprocessed/` is committed (used on another device).
- Gripper attachment: `geometry.tool_box` (x/y/z [min,max] mm in `flange` or `tool` frame) = 8 corner chain points 9..16 checked as zero-radius capsules named `tool_box` (exact vs planar walls; end-effector + pose-determined). `chain_points_batch` now returns (N,17,3). Base may sit below the floor (base column never floor-checked; static check only refuses if the shoulder housing is in the floor). Overlay draws tool prism + box on Studio's GROUPS[7] with a measured fixed matrix.
- Fit clean copies carry `fit_version` (`fit.FIT_VERSION`); bump it when the fitting algorithm changes so old copies are refit. The approach leg (ready -> start joints) is a pass/fail constraint, not part of the maximized slack.
- Workspace map for the current cell: `outputs/workspace/dexterity_map.png` (2026-10-14): tool-down feasible for flange height ~250-800 mm above the floor; best dexterity/orientation range ~300-650 mm, x 100-450 mm.
- Standard xArm Gripper = `tool_box` x ±37.5, y ±53.12, z 0..175.86 (flange frame, from Studio's xarm_gripper.stl; frame verified in sim) with `radii_mm.tool: 0`. Sim end tool can be set with `arm.set_collision_tool_model(1)` (Studio then draws the gripper). The user edits config/safety.yaml in their IDE: their saves can overwrite edits made while it's open, so re-read it before relying on it.
- Workcell can be given as `workcell.walls_in_base_mm: {x_min..z_max}` (base frame, walls parallel to base axes; converted in `config.walls_in_base_to_box`; same fingerprint as the equivalent box form). `scripts/control/measure.sh where|watch|walls|wall-test` (`safety/measure.py`): pose/gripper-model report, live clearance, touch-off wall measurement (prints the YAML block), slow virtual-wall stop test (`--direction y+` etc.; confirms before moving, returns to start). Live overlay on any Studio: `python -m src.xarm_pipeline.sim_scene_overlay --url http://<ip>:18333`.
- Self-collision: `Workcell.body_slack` = end effector vs base column + gripper vs upper_arm/elbow/elbow_offset/forearm; in preprocess (reject), runtime monitor (closing-rate lookahead), fit and test generator. Firmware C22 = self-collision (keep it enabled on the real robot with the right end-tool model). `runtime.predict_monitor_margin(wc, cfg, t, Q)` predicts monitor stops from a plan.
- Real-robot tests: `python -m src.xarm_pipeline.safety.test_trajectories corners|dynamic` -> `data/preprocessed/NNN_*/tests/`. corners: tip (box far-face centre) as close as allowed, free orientation, corner-to-corner tour as a JOINT trajectory (play with `playback --control servo`, new timed joint streaming). dynamic_tests.csv dyn_{x,y,z}_{20,50,100}pct (Cartesian, play_cartesian_ik).
- Recorded mocap: `scripts/control/prepare_recorded.sh <dir> <scale> <num...>` (`safety/recorded.py`): time-scale, cut over-limit spans (NOTICE), fit/split, joint-limit check, ease part ends, densify to 100 Hz -> `data/preprocessed/NNN_*/recorded/<slate>/<num>_t<scale>.csv`. ~2-2.5x slowdown needed for the 5ec359d2 slate.
- Servo streaming needs dense setpoints (<= 10-20 ms); per-command check uses the validated joints (fast path). The sim firmware hangs after many runs ("failed to check version") -> `docker restart uf_software`, start firmware, `set_collision_tool_model(1)`, move to ready.
- Review videos: `outputs/review_videos/` (2026-10-14).
- Safety demos: `run_safety_demos.sh --config tests/fixtures/safety_test.yaml` (designed for that cell; always `--no-fit`).
- Firmware fence state if a run was killed: `scripts/control/firmware_fence.sh status|reset`.
- Demos: `scripts/control/run_safety_demos.sh` (~6 min, needs sim) → `outputs/safety_demo/<run>/` (sim .webm, wall plot, animation, side-by-side) + `demo_summary.json`.
- Tests: `.venv/bin/python -m pytest tests -q` (33 tests, no robot). They use `tests/fixtures/safety_test.yaml`, not the live config, so real measurements don't break them.
- Caveats: sim stops near-instantly, real arm won't — `latency_s`/`stopping_time_s` must be measured on hardware. Monitor reaches ~190–246 Hz (target 250) due to GIL contention. Not covered: held objects, cables, in-box obstacles, self-collision. With the default 172 mm tool, the bundled `sample_sine_joint.csv` (starts at all-zeros) is rejected — tool would be below the floor.
- Keep all layers enabled for anything that moves the real arm.

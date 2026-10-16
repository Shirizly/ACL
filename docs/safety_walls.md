# Safety virtual walls (xArm 7)

The arm is kept inside a rectangular workcell box. The box is enforced in three
independent layers, and each layer checks the **whole arm** (links + tool), not
only the TCP:

| Layer | When | What it checks | What it does | Disable (demo only) |
|---|---|---|---|---|
| 1. Preprocessing | before anything moves | every sample + dense interpolation, whole arm | `reject` the trajectory, or `clip` TCP targets | `--disable-safety-preprocess` |
| 2. Runtime guard | during playback | (a) every command before it is sent, (b) a monitor thread polling measured joints at 250 Hz with velocity lookahead | stops the arm (`set_state(4)`) and aborts playback | `--disable-runtime-guard` |
| 3. Firmware fence | during playback | firmware TCP **point** vs an axis-aligned base-frame box | the controller stops the arm (error C35) | `--disable-firmware-fence` |

All three are **on by default** in `playback.py` and `cartesian_playback.py`.
Disabling preprocessing does **not** disable the runtime guard, and the reverse
is also true. Every disable prints a loud `!!!` banner and is recorded in
`safety_run_summary.json`.

Code: `src/xarm_pipeline/safety/`, config: `config/safety.yaml`, tests:
`tests/test_safety.py`.

---

## 1. Frames

```
            world / floor frame ("box frame")                     robot base frame
                                                                  (what get_position() reports)
   z (up)                                              
   ^      ceiling z = H                                        p_world = Rz(yaw) * p_base + (bx, by, bz)
   |   +---------------------------+                       
   |   |                           |                             top view (looking down -z):
   |   |          ___              |                         
   |   |         |arm|             |                          y ^   wall +y  (y = +W/2)
   |   |         |   |             |                            | +-----------------------+
   |   |        _|___|_            |                            | |                       |
   |   |       | base  |  (bx,by,bz)                          | |      x_base           |
   +---+-------+-------+-----------+----> x                    | |       ^  /             |
  floor z = 0     origin = center of the floor                  | |  yaw  | /             | wall +x
                                                               --+-|-------o-------------- |--> x
   x in [-D/2, +D/2]  (D = depth,  along world x)                 | |     base (bx,by)      |  (x = +D/2)
   y in [-W/2, +W/2]  (W = width,  along world y)                 | |                       |
   z in [0, H]        (H = height)                                | +-----------------------+
                                                                  |   wall -y  (y = -W/2)
```

* **World/box frame**: origin at the center of the box floor, z up. D runs along
  world x, W along world y, H along z.
* **Robot base frame**: the frame of `get_position()` and of Cartesian
  trajectory CSVs. Its pose in the world frame is `robot_base_in_world`
  (x, y, z, yaw about world z). Every check transforms robot points into the
  world frame first (`Workcell.base_to_world`).
* Walls are named `wall -x`, `wall +x`, `wall -y`, `wall +y`, `floor`, `ceiling`
  in the world frame.

## 2. Configuration (`config/safety.yaml`)

All lengths are in **mm**, angles in **deg**, and times in **s**. Values marked
`[MEASURE]` are placeholders and must be measured on the real cell.

| Field | Default | Meaning |
|---|---|---|
| `workcell.depth_D_mm / width_W_mm / height_H_mm` | 1200 / 800 / 1100 | box size; walls at x = ±600, y = ±400, z = 0 and 1100 |
| `robot_base_in_world.{x,y,z}_mm, yaw_deg` | 0, 0, 0, 0 | base pose in the box frame; `z_mm` may be negative (base below the floor, see §3) |
| `workcell.walls_in_base_mm.{x,y,z}_{min,max}` | — | alternative to the two rows above: each wall's coordinate along a robot-base axis (walls parallel to the base axes). Measure it with `measure.sh walls`. |
| `safety_margin_mm` | 20 | clearance every capsule must keep from every wall |
| `geometry.tool_offset_mm` | [0, 0, 172] | tool tip in the flange frame (xArm Gripper TCP = 172 mm) |
| `geometry.radii_mm.*` | 50–80 | capsule radius per link (see §3) |
| `geometry.tool_box` | none | extra rigid box on the gripper: `frame` (`flange` or `tool`), `x_mm`/`y_mm`/`z_mm` as [min, max], optional `inflate_mm` (see §3) |
| `geometry.firmware_tcp_offset` | zeros | firmware TCP offset (used offline; when connected, the robot's value is read) |
| `preprocess.enabled / policy` | true / reject | `reject` or `clip` |
| `preprocess.interp_resolution_mm / _deg` | 5 / 1 | maximum motion of any arm point between checked configurations |
| `preprocess.ik_model_tolerance_mm` | 10 | extra clearance for links whose position depends on IK redundancy (Cartesian only) |
| `preprocess.clip_smoothing_s` | 0.3 | clip offsets are smoothed over about this time (never reduced) so the clipped path decelerates |
| `runtime.enabled / command_check` | true / true | runtime guard and its per-command check |
| `runtime.monitor_rate_hz` | 250 | monitor polling rate (it has its own TCP connection) |
| `runtime.stop_margin_mm` | 5 | the monitor stops when predicted clearance < this |
| `runtime.latency_s + stopping_time_s` | 0.02 + 0.08 | lookahead horizon. **Measure these on the real robot.** |
| `firmware_fence.enabled / shrink_mm / restore_on_exit` | true / auto / true | see §4.3 |

**Why these defaults.** The xArm 7 reaches about 700 mm from the shoulder (the
shoulder is 267 mm above the base). With the base centered:
* The y walls at ±400 mm sit well inside the reach. The flange can only go to
  about |y| ≈ 330 mm.
* The x walls at ±600 mm are reachable with an outstretched arm.
* The floor is reachable by the 172 mm gripper.
* The ceiling at 1100 mm is reachable only with the arm straight up.

The elbow can also reach a y wall while the tool is far away (see the elbow
demo). These defaults are meant to make tests hit walls; they are not a real
cell.

**Consequences of the defaults that you will notice:**
* `data/trajectories/sample_sine_joint.csv` starts at the all-zero home pose.
  There the flange is at z = 120.5 mm pointing down, so a 172 mm gripper would
  be 51 mm *below* the base plane. This trajectory is **rejected**.
* `sample_cartesian_circle.csv` (flange z 240–260 mm) brings the gripper within
  18 mm of the floor, so it is **rejected** under `reject` and lifted by up to
  5 mm under `clip`.

If no gripper is mounted, set `tool_offset_mm: [0, 0, 0]`.

## 3. Geometry model

* **FK**: `safety/kinematics.py` holds the xArm 7 modified-DH parameters. They
  were read from the firmware (`arm._arm.get_dh_params()`: 7 × (θ-offset, d, α, a)):
  d = 267, 0, 293, 0, 342.5, 0, 97; a = 0, 0, 0, 52.5, 77.5, 0, 76;
  α = 0, −90, 90, 90, 90, 90, −90 deg.
  * Validation: against `arm.get_forward_kinematics()` on 500 random joint
    vectors in the simulator, the worst error was **0.00016 mm** in position
    and **0.00003°** in orientation. The DH difference vs the firmware was
    4e-8.
  * `test_fk_matches_firmware` pins four firmware reference poses.
  * Real robots ship with per-unit calibrated DH. Compare them with
    `python -m src.xarm_pipeline.safety.validate --trajectory <any csv> --ip <ip> --check-robot-dh`.
* **Capsules** (segment + radius) between chain points:

```
 base (static) : floor(0,0,0) -> shoulder O1 (z=267)        [checked once at config load, side walls/ceiling]
 upper_arm     : shoulder O1 -> elbow O3
 elbow         : O3 -> O4 (= O3 + 52.5 x3)
 elbow_offset  : O4 -> O4 + 77.5 x4 (forearm start)
 forearm       : forearm start -> wrist O5
 wrist         : O5 -> O6 + 76 x6
 wrist_offset  : O6 + 76 x6 -> flange O7
 tool          : flange -> flange * tool_offset_mm
 tool_box      : 8 corners of geometry.tool_box, rigid with the flange (radius = inflate_mm, default 0)
```

* **Clearance** between a capsule and an axis-aligned wall is
  `min(dist(end A), dist(end B)) − r`. This is exact because a half-space is
  convex. A capsule is valid when its clearance ≥ the required value for all
  six walls.
* **The radii are guesses** that enclose the xArm 7 outline (link diameter
  about 100–130 mm). Measure them, especially for the tool. Anything you hold
  in the gripper is **not** modelled; enlarge `tool_offset_mm` / `radii_mm.tool`
  to cover it.
* **Gripper attachment (`geometry.tool_box`).** A box fixed to the gripper, given as [min, max] ranges along
  the axes of `frame`. `flange` is the J7 flange face, with z pointing out toward the gripper; these are the same
  axes as the firmware TCP frame with zero TCP offset. `tool` is the same axes moved to `tool_offset_mm`. Against a
  flat wall a box's clearance is its nearest corner's, so the 8 corners are checked as zero-radius points.
  That's exact, not an approximation. The box counts as end-effector geometry: Cartesian `clip` and the fit move
  it with the target, and it needs no IK-model tolerance. It is checked *in addition to* the round `tool`
  capsule; shrink `radii_mm.tool` if the box now describes the gripper body. With `record_web_sim --walls`, the box
  (yellow) and the tool capsule (gray prism) are drawn on the arm in the sim (`visualization.show_tool`). The
  overlay's flange-to-model transform was measured in the sim and is constant across poses to 0.2 mm.
* **Standard xArm Gripper (AG1002).** Model it with `tool_box` and set `radii_mm.tool: 0`:
  `frame: flange, x_mm: [-37.5, 37.5], y_mm: [-53.12, 53.12], z_mm: [0, 175.86]` (y = jaw direction). This is the
  exact bounding box of UFACTORY's own mesh (`static/stl/xarm_gripper.stl` in Studio). Reading the mesh back from
  the sim's 3D view at two poses showed that Studio mounts it with its file axes equal to the flange frame (0.2 mm).
  The earlier round model (radius 50 mm) was 3 mm too narrow along y and missed the body corners by up to 15 mm.
  The 3D gripper Studio draws, and the firmware's own end-effector model (`set_collision_tool_model`, used only for
  the arm-hits-itself check), are not used by these wall checks: only this config is.
* **Base below the floor.** The base column is never checked against the floor, so `robot_base_in_world.z_mm` may
  be negative and the base may protrude through it. Every moving link is still checked against the floor,
  including the upper arm from the shoulder joint up. The config is refused only if the shoulder housing itself
  would be in the floor: `z_mm < -(267 - radii_mm.upper_arm - safety_margin_mm)`, about −182 mm with the defaults.

## 4. Layers

### 4.1 Preprocessing (`safety/preprocess.py`, CLI `safety.validate`)

**Joint trajectories** (`t,q1..q7[,dq]`, rad):
* FK of every sample, plus the joint-linear segment from the robot's current
  joints to sample 0 (the approach move), plus each segment between samples.
* Segments are subdivided so that `Σ|Δq_i| · R_max ≤ interp_resolution_mm`,
  where R_max ≈ 1.24 m bounds the distance from any joint axis to any arm
  point. No point of the arm can therefore move more than 5 mm between two
  checked configurations.
* Half the resolution (2.5 mm) is added to the required clearance, so a
  crossing **cannot hide between samples**. The `test_dense_interpolation_bound`
  and `test_crossing_between_samples_is_detected` tests cover this.
* Velocity mode also checks the integrated `dq` path.
* Joint trajectories can only be rejected, never clipped.

**Cartesian trajectories**: the firmware's IK picks the joints. Measured
behaviour of `get_inverse_kinematics` (sim firmware v2.4.0):
* It is seeded from the robot's **current** joints. `ref_angles` needs firmware
  ≥ 2.7.103.
* The same pose gave very different solutions depending on where the robot
  was. From the singular all-zero home it gave J4 = 64.6°, J7 = −134°.

It therefore cannot predict a trajectory's joints offline. Instead we use a
local damped-least-squares IK run as a *continuation* along the densely
interpolated Cartesian path (≤ 5 mm, ≤ 1° steps):
* It starts from the robot's actual joints, or an offline seed / `--seed-joints-deg`.
* If any arm point jumps more than 5 mm between steps, the step is bisected.
* While the simulator streamed a large servo-Cartesian sweep, this prediction
  matched the firmware's actual joints to **0–4.7 mm on every chain point**,
  with slow drift.
* The pose-determined capsules (`wrist_offset`, `tool`) need no IK margin.
  The other links get `ik_model_tolerance_mm` = 10 mm extra.
* **Limitation**: near singularities (e.g. starting from the all-zero home)
  the firmware and the local IK can choose mirror-image solutions, which we
  observed: J1 +33° vs −29°. Always validate from the real start state; the
  playback integration does this automatically. The runtime guard covers the
  rest.

**Policies**:
* `reject`: refuse the whole trajectory. The report gives the first violating
  sample (and the segment it is in), the time, the link, the wall, the
  clearance, the required clearance, the deficit and the world point.
* `clip`: for each sample, translate the commanded position (orientation kept)
  so the end-effector capsules (wrist, wrist_offset, tool) are back inside the
  box shrunk by margin + resolution/2 (+ IK tolerance for the wrist). This
  iterates with IK.

  A hard clip would turn a 220 mm/s approach into an instant stop at the
  boundary. In the first demo run that made the servo tracking overshoot to
  333 mm/s and tripped the monitor (a false stop), and it is dynamically
  impossible for a real arm. The per-axis offsets are therefore smoothed:
  dilate (running max over ±`clip_smoothing_s`), then apply two box filters.
  The result is ≥ the hard offset at every sample, so it is never less safe,
  and it has continuous velocity. Peak acceleration on the demo path fell from
  4335 to 738 mm/s².

  The clipped trajectory is then re-checked in full; if any other link
  (forearm, elbow, upper arm) still violates, the trajectory is **rejected**.
* Velocity-mode trajectories are reject-only.

**CLI** (no robot needed for joint CSVs):
```bash
python -m src.xarm_pipeline.safety.validate --trajectory data/trajectories/x.csv \
    [--policy clip] [--ip 127.0.0.1] [--report r.json] [--sanitized-out clipped.csv]
scripts/control/validate_trajectory_safety.sh data/trajectories/x.csv clip    # XARM_IP=... to seed from robot
```
Exit codes: 0 = ok/clipped, 2 = rejected, 3 = config error.

**In playback**, the check runs after connecting, using the current joints and
TCP offset, and the robot is not moved if the trajectory is rejected (exit code
2). `--safety-policy clip` executes the clipped trajectory and saves it as
`safety_sanitized_trajectory.csv`. `scripts/control/prepare_cartesian_start.sh`
also checks its approach move (`XARM_SAFETY_PREPROCESS=0` disables that).

### 4.2 Runtime guard (`safety/runtime.py`)

* **Per-command check** (`CommandGuard`), before each send, requiring clearance
  ≥ `safety_margin_mm`:
  * Joint targets: dense FK of the segment from the previous target.
  * Cartesian targets: local IK seeded from the latest *measured* joints
    (0.3 ms per step), then a full-arm check.
  * Joint velocity: `q_now + dq·(dt + lookahead)`.
  * Cartesian velocity: the end-effector capsules translated by
    `v·(dt + lookahead)`.
  * Measured check time: see the demo results.
* **Monitor thread** (`SafetyMonitor`):
  * Uses its **own XArmAPI connection**, so the stop never waits for the
    playback socket.
  * Polls `get_servo_angle` at 250 Hz (a real request; about 0.1 ms round trip
    to the local sim).
  * Computes FK of all capsules and each point's velocity over a 20 ms window.
  * Computes predicted clearance = clearance − max(0, outward speed) ×
    (latency_s + stopping_time_s) per wall.
  * Stops when predicted clearance < `stop_margin_mm`.
  * Also stops if joint feedback is lost for > 100 ms.
* **Report rates measured** (SDK `register_report_callback`):
  * The default `rich` / `normal` report sockets push at only **5 Hz** (200 ms),
    which is useless for this.
  * The `report_type='real'` socket (port 30003) pushes at **100 Hz**.
  * Direct polling is faster still, which is why the monitor polls (and opens
    its connection with `report_type='real'`).
* **How to stop, measured in the sim** (300 mm/s linear move or 200 mm/s
  Cartesian velocity, stop sent from a second connection):

| Method | Call | Arm halted | Travel after call |
|---|---|---|---|
| `set_state(4)` | 0.4–1.5 ms | < 1 report period (10 ms) | 0.8–1.2 mm |
| `emergency_stop()` (loops `set_state(4)`) | 1 ms | < 10 ms | 0–1.2 mm |
| `set_state(3)` (pause) | 1 ms | < 10 ms | 0 mm |
| zero Cartesian velocity (mode 5) | 0.5 ms | **369 ms** | **32.6 mm** |

  The guard sends `set_state(4)`, then a zero velocity in modes 4 and 5, then
  `emergency_stop()` to confirm state 4. Commands the playback thread sends
  afterwards are rejected by the controller (state 4). Playback aborts with
  `playback ABORTED by safety: ...` and exit code 3. Logs are written to
  `safety_monitor_log.csv` and `safety_monitor_summary.json`.
* **Measured in the demos:**
  * The monitor achieved **177–245 Hz** (target 250 Hz). Single loop
    iterations reached up to 30 ms while the per-command IK check ran: both
    share the Python GIL.
  * The per-command check took up to 5–37 ms worst case for Cartesian targets
    (dense IK when the robot lags behind the command).
  * `trip()` took 3.6–15 ms from `set_state(4)` until state 4 was confirmed by
    `emergency_stop()`.
  * Budget for all of this in `latency_s`. Moving the monitor into its own
    process would remove the GIL contention.
* **Joint position mode queues motions.** `set_servo_angle(wait=False)` in
  mode 0 queues one planned move per command, and the robot runs about 2× the
  CSV time behind. The per-command check therefore refuses a bad target long
  before the arm gets there. In the elbow demo it stopped with the arm still
  100 mm from the wall. This is safe, but the stop is earlier than "at the
  wall".
* **Sim quirk.** In servo mode the simulator's reported joints change in steps
  (one jump per command), so the finite-difference speed is spiky. You can see
  this in the "predicted" curve of the plots.

  **The simulator stops instantly; a real arm does not.** Measure
  `stopping_time_s` (and the stopping distance at your speeds and payload) on
  the real robot and update the config.
* **Recovery** after a stop: `clean_error()`, `clean_warn()`,
  `motion_enable(True)`, `set_mode(0)`, `set_state(0)`, then move away from the
  wall (`safety.demo.recover_and_home`).

### 4.3 Firmware fence (`safety/firmware_fence.py`)

Behaviour measured with SDK 1.18.4 and sim firmware v2.4.0:
* `set_reduced_tcp_boundary([x_max,x_min,y_max,y_min,z_max,z_min])` (integer
  mm, **robot base frame, axis-aligned**) plus `set_fence_mode(True)`. Read it
  back with `get_reduced_states()`.
* Only the firmware **TCP point** is checked. The elbow went 137 mm past a
  z_max boundary without triggering it.
* It **stops, it does not clamp**:
  * `set_position` / `set_servo_angle(wait=True)` toward the outside move up to
    the boundary and return −9, with state 4 and no error code.
  * Servo mode (1) and joint/Cartesian velocity modes (4/5) raise
    **ControllerError C35** at the boundary (stopped at 400.2–401.1 mm for a
    400 mm limit).
  * Recover with `clean_error` + `motion_enable` + `set_state(0)`.
* Settings are not persisted: we never call `save_conf()`, and after a
  simulator restart the boundary was ±9999 with the fence off.

How it is used:
* The fence box is the largest base-frame axis-aligned box inscribed in the
  (possibly yawed) world box shrunk by `shrink_mm`.
* `auto` = margin + tool radius = 70 mm, giving x ±530, y ±330, z 70…1030 for
  the defaults.
* Any configuration that passes layers 1/2 satisfies this for the flange point,
  so the fence never fires before the software layers on a valid trajectory.
  It is a **coarse backstop** for software failure (crash, bypass), not a
  full-arm guarantee.
* It is set at playback start and restored to the previous values at exit.
* If a process is killed in between, check and reset by hand:
  ```bash
  scripts/control/firmware_fence.sh status      # or: python -m src.xarm_pipeline.safety.firmware_fence --ip <ip> status
  scripts/control/firmware_fence.sh reset       # fence off, boundary +-9999
  ```

## 5. Guarantees and limits (read before using the real robot)

**Guaranteed, given a correct config and a correct kinematic model:**
* A trajectory that passes preprocessing keeps every capsule ≥ `safety_margin_mm`
  from every wall:
  * at every sample, along joint-linear segments (joint CSVs), and along the
    linear/slerp Cartesian path (Cartesian CSVs);
  * for Cartesian CSVs, the non-end-effector links are covered only up to the
    IK prediction tolerance.
* While playing, no command whose target would violate the margin is sent.
* The arm is commanded to stop when measured geometry is predicted to come
  within `stop_margin_mm` within the lookahead horizon.

**Not guaranteed or not covered:**
* **Stopping distance.** The runtime guard is only as good as `latency_s` +
  `stopping_time_s`.
  * In the sim the arm halts within one 10 ms sample.
  * A real xArm decelerates; at 500 mm/s and 0.1 s that is about 50 mm.
  * The lookahead uses the speed *toward* the wall. A valid trajectory that
    approaches a wall fast and brakes late can therefore trip the monitor
    (false stop). This is conservative, not unsafe.
* **Geometry model errors.**
  * Radii are estimates.
  * Real DH parameters deviate slightly from nominal.
  * Cable dress, the camera, **objects held in the gripper**, and gripper
    finger opening are not modelled.
* **Cartesian IK prediction.**
  * It is approximate (measured ≤ 4.7 mm in sim, 10 mm tolerance applied).
  * It can differ entirely near singularities or from a different start
    configuration.
  * Validate with the real start state; the runtime guard checks measured
    joints.
* **Only the 6 box walls.** There are no internal obstacles, no self-collision
  check, and no human detection. This is not a certified safety function: it
  does not replace the robot's own safety system or a risk assessment.
* **Firmware fence**: TCP point only, base-frame axis-aligned, conservative box
  for a yawed base.
* **Monitor dependencies.** The monitor needs network feedback. If feedback is
  lost it tries to stop the arm, but if the network is gone that stop command
  cannot arrive either.
* **Sim vs real**:
  * Report rates and polling latency are measured over localhost.
  * The sim has no dynamics or payload, and its stop is instantaneous.
  * The firmware version differs (`ref_angles` IK seeding exists only on
    ≥ 2.7.103).

**When entering real measurements:**
1. Measure the box: D, W, H, and the floor level the base stands on.
2. Measure the base pose in the box: x, y, z, yaw. A yaw error of 1° moves a
   point 700 mm away by 12 mm, so keep the margin larger than your measurement
   error.
3. Set the tool geometry (`tool_offset_mm`, `radii_mm.tool`), and set
   `firmware_tcp_offset` to match Studio.
4. Run `--check-robot-dh`.
5. Measure the stopping time and distance: run a demo with the guard ON at your
   speeds and look at `safety_monitor_log.csv`. Set `latency_s` and
   `stopping_time_s` from that.
6. Run `pytest tests -q`, then validate your trajectories offline before
   running them.

## 6. Demos (simulator)

```bash
scripts/sim/up.sh
docker exec -d uf_software /xarm_scripts/xarm_start.sh 7 7   # non-interactive firmware start
scripts/control/run_safety_demos.sh --config tests/fixtures/safety_test.yaml     # all runs, ~6 min
scripts/control/run_safety_demos.sh --config tests/fixtures/safety_test.yaml --only tcp_wall_runtime_guard
```

The demo trajectories were designed for the original test cell, so pass `--config tests/fixtures/safety_test.yaml`.
With another box they still run, but they may not cross or approach the walls as described. The demos always play the
raw trajectories (`--no-fit`), never fitted clean copies.

Demo trajectories are generated to `data/trajectories/`:
* `safety_demo_tcp_through_wall.csv`: the flange, pointing down at
  (300, 0, 300), goes to y = 560 and back. The tool passes 200 mm through the
  +y wall.
* `safety_demo_elbow_through_wall_joint.csv`: J1 = 90°, J4 = 60°, J6 = 120°;
  J2 swings −40° → −100° → −40°. The elbow goes about 30 mm through the −y wall
  while the tool stays ≥ 96 mm inside.

Each run writes `outputs/safety_demo/<run>/` containing:
* `<run>_sim.webm`: headless Studio recording, camera rotated/zoomed via
  `record_web_sim.py --orbit-drag 120,60 --zoom-steps 4`;
* `safety_walls_plot.png`: top + two side views of the measured tool tip,
  elbow and flange vs the walls and margin, plus clearance vs time;
* `safety_walls_anim.mp4` and `<run>_side_by_side.mp4`: sim | animated wall
  view, aligned by wall-clock time. The video start is estimated as
  `video_end_epoch − duration`; the residual misalignment is about 0.5 s.
  `record_web_sim.py --stop-when-done` ends the recording 1.5 s after
  playback exits;
* the monitor log and summary, the preprocessing report, the run log, and
  `demo_result.json`.

`outputs/safety_demo/demo_summary.json` collects the minimum wall distances and
stop or reject outcomes. The web sim cannot draw the walls, so read the plots
together with the video.

### Demo results (sim, default config, recorded 2026-10-13)

"Min clearance" is the measured minimum distance from any capsule surface to
any wall over the run. It comes from monitor-polled joints + FK; negative
means through the wall.

| Run | Layers on | Outcome | Min clearance (link / wall) |
|---|---|---|---|
| `tcp_wall_no_safeguards` | none | ran to completion; tool went through the wall | **−210.7 mm** (tool / +y) |
| `tcp_wall_preprocess_reject` | all, policy=reject | rejected before moving (exit 2): first violation sample 106 (t=3.18 s), `wrist` vs +y, 30.6 mm < 32.5 mm required | robot not moved |
| `tcp_wall_preprocess_clip` | all, policy=clip | 180/301 samples shifted (max 243 mm), executed fully, no stop, fence not triggered | **33.0 mm** (wrist / +y) |
| `tcp_wall_runtime_guard` | runtime guard only | monitor stop at actual 38.1 mm, 332 mm/s (predicted 4.9 mm < 5 mm) | **38.1 mm** (tool / +y) |
| `tcp_wall_velocity_monitor_only` | monitor only (Cartesian velocity mode, no command check) | monitor stop at actual 24.1 mm, 217 mm/s | **23.3 mm** (tool / +y) |
| `tcp_wall_firmware_fence_only` | firmware fence only | controller error C35 when the flange reached y≈330 (fence box) | **13.6 mm** (tool / +y): did not cross the wall but came inside the 20 mm margin, since the fence checks only the flange point |
| `elbow_wall_no_safeguards` | none | ran to completion; elbow went through the wall, tool stayed ≥ 96 mm inside | **−30.5 mm** (elbow_offset / −y) |
| `elbow_wall_preprocess_reject` | all | rejected before moving: sample 74 (t=3.70 s), `elbow_offset` vs −y | robot not moved |
| `elbow_wall_runtime_guard` | runtime guard only (joint position mode) | command 74 refused (target 19.8 mm < 20 mm); the arm was still behind its queued moves | **102.9 mm** (wrist / +y) |
| `elbow_wall_runtime_guard_velocity` | runtime guard only (joint velocity mode) | command 72 refused (velocity prediction 150 ms ahead) | **28.2 mm** (elbow_offset / −y) |

Additional measurements:
* Stop command → state 4 confirmed: 1.7–6 ms.
* Monitor rate: 190–246 Hz.
* Worst per-command check: 7 ms (joint) / 33 ms (Cartesian, while the target
  was far outside the box).
* After each stop the demo recovers the sim (`clean_error`, `motion_enable`,
  `set_state(0)`, fence reset) and moves to the safe home
  q = (0, −28.17, 0, 23.24, 0, 51.41, 0)°.

## Seeing the walls in the sim video

`record_web_sim` can draw the workcell into the Studio 3D view and record only the arm:

```bash
python -m src.xarm_pipeline.record_web_sim --headless --arm-only --walls --seconds 30 --stop-when-done 1.5 \
  --output-dir outputs/videos --run-command "scripts/control/play_cartesian_ik.sh data/trajectories/x.csv servo"
# camera: --view-azimuth 35 --view-elevation 25 [--view-zoom Z]   other config: --safety-config path.yaml
```

- `--walls` draws the box walls (translucent faces + edges), a faint inner box at `safety_margin_mm`, and any
  `visualization.objects` from the safety YAML (tables, bins, fixtures: boxes in the world/floor frame).
- These objects are **visual only**. The simulator firmware does not know about them, and the safety checks only
  use the workcell box. An object drawn in the video is not avoided.
- How it works: Playwright injects three.js meshes into Studio's own scene (`Arm3D` Vue component →
  `vm.three.scene`; scene units are 100 mm, Y up, base at y = −5). Nothing in the simulator container is modified.
  A Studio update that renames this component would break the overlay; the recorder then prints a message and records normally.
- The Studio model has no gripper, so the tool capsule that the safety checks use is not drawn.

## Fitting trajectories into the cell (clean copies)

Only the *relative* motion of a Cartesian trajectory is assumed to matter. Before execution, `safety.fit` re-places
each trajectory so it fits the workcell, and stores a clean copy in the **preprocessed set** for the current safety
setup (next section). This happens automatically when you play a Cartesian trajectory. To prepare clean copies ahead
of time, for example on a fast machine before moving to the robot's computer:

```bash
scripts/control/fit_trajectories.sh data/trajectories/bank.csv           # -> data/preprocessed/NNN_*/trajectories/bank.csv
scripts/control/fit_trajectories.sh "data/trajectories/bank.csv#pick_1"  # one trajectory of a grouped file
scripts/control/fit_trajectories.sh data/trajectories/                   # every CSV in the directory
```

**What may change** (options in the `fit:` section of the safety YAML):

| Parameter | Effect | Default |
|---|---|---|
| translation | moves the whole trajectory; always searched | on |
| `yaw` | rotates the whole trajectory (positions *and* orientations) about the vertical axis through its center; relative motion is kept exactly | ±180° |
| `spin` | rotates the tool about its own approach axis, orientations only (fine for a symmetric gripper; changes the angle between tool and direction of motion) | 0 (off) |
| `tilt` | tilts the whole trajectory relative to gravity | 0 (off) |

Why not "add a constant to roll/pitch/yaw"? Adding to `yaw` *is* a rotation about the base vertical, so it's the
orientation half of the rigid yaw above. Adding to `roll` is a rotation about the tool x-axis. Adding to `pitch` is not
a constant rotation at all. A rotation applied to positions and orientations together is the only offset that keeps
"relative movement" meaningful.

**How it searches.** Each candidate placement is scored with the same full-arm model as the safety preprocessing:
continuation IK from the ready pose, then capsules against the walls. The score is the worst *slack*: clearance minus
the required clearance, in mm. An analytic bound on the flange and tool tip skips hopeless candidates without running IK.
The search is lexicographic: first reach `good_enough_slack_mm`, then prefer placements near
`preferred_center_base_mm` (in front of the arm) and with the least rotation. Without that preference, maximizing
clearance pulls everything to the middle of the box, which is over the base column. Seeds come first from the mean of
earlier placements in `fit_log.jsonl`, which is kept per workcell/tool configuration fingerprint. Then a pattern search
refines the result. Expect about 15–45 s per trajectory in a roomy cell and several minutes when it has to split.

**Splitting.** If no placement fits, the trajectory is cut near its middle (slow samples preferred, both halves
pre-screened with the analytic bound). Each half is then fitted independently, recursively, down to `min_part_s` and
`max_depth` levels. Parts share the cut sample, so no motion is lost. Parts that still don't fit are **dropped** and
listed in the report. Output names: `name__p1`, `name__p2`, … (a grouped file).

**Outputs.**
- `x.csv`: clean trajectories, always with explicit `roll,pitch,yaw` (unwrapped) and twist columns rotated
  along with the placement. It is grouped (`traj` column) if the input was grouped or got split.
- `x.fit.json`: the source file's SHA-1, and per trajectory/part: the source rows and times, the parameters
  (`cx,cy,cz` world center, angles), the world shift, the slack, the dropped parts, and the validated approach
  (`approach_via_joints_deg`, `start_joints_deg`).
- `fit_log.jsonl` (per set): every accepted placement, used to seed the next search.

**Approach (why clean copies carry joint angles).** The arm has 7 joints for a 6-DOF pose, so the same tool pose can be
reached with the elbow in different places. The fit's full-arm check assumes a particular elbow position at the start.
A straight-line Cartesian approach planned by the firmware can leave the elbow somewhere else. In a sim test this made
playback predict the elbow 2.3 mm through a wall on a part the fit had accepted with 30 mm to spare. So for a clean copy,
`prepare_cartesian_start.sh` does two **joint moves** instead:
1. To the fit's seed configuration (the ready pose, `fit.seed_joints_deg`).
2. To the exact start configuration the fit validated.

The fit includes the second leg in its clearance score, and the prestart check validates both legs from the robot's
actual joints before moving. It then confirms the tool reached the trajectory start (within 2 mm; otherwise the robot's
TCP offset probably differs from `geometry.firmware_tcp_offset`). Requiring a safe approach from the ready pose
constrains the fit a little more: in the test, a 2000 mm line split into 4 parts instead of 2.

**Limits.**
- The fit is checked offline from the ready pose (`seed_joints_deg`). Playback still runs every safety layer from the
  robot's real start state, including both approach legs.
- The local IK resolves the arm's redundancy only approximately like the firmware does. That's why there's extra
  required clearance on the elbow and forearm.
- Joint-space trajectories are skipped: absolute joint angles can't be re-placed.
- Each split part has its own placement, so parts don't join up in space.

## Preprocessed sets (one per safety setup)

Clean copies depend on the setup they were fitted for, so they live in one directory per setup. Several box sizes or
tool configurations can then coexist, and the data can be committed and used on another machine:

```
data/preprocessed/
  001_D1200xW800xH1100/
    safety.yaml            frozen copy of the safety config the set was created with
    meta.json              index, fingerprint, creation time, box / base / tool summary
    fit_log.jsonl          placements found in this set (seeds later fits)
    trajectories/bank.csv  clean copy of data/trajectories/bank.csv (+ bank.fit.json)
    external/x_1a2b3c4d.csv  clean copy of a file from outside data/ (e.g. another repo)
  002_D600xW500xH1100/
    ...
```

**Matching.** A set is matched by a **fingerprint of the setup**, not by comparing files: `workcell`,
`robot_base_in_world`, `safety_margin_mm`, `geometry` (tool, link radii, firmware TCP offset), the IK/interpolation
tolerances in `preprocess`, and the whole `fit:` section. Numbers are normalized (`1200` = `1200.0`). Comments, `runtime`
timing, `firmware_fence`, `visualization` and the reject/clip policy don't matter, so tuning them reuses the existing set.
Change any setup value and the next execution creates a new set, `NNN+1_…`. The directory label shows only D×W×H.
Sets with the same box but a different base pose or tool get different numbers; `meta.json` has the details.

**At execution** (`cartesian_playback`, `play_cartesian_ik.sh`, `play_cartesian_ik_record.sh`):
1. Find the set for the current config, or create it.
2. Look for the clean copy at the mirrored path. Refit it if it's missing, if the source file changed (SHA-1), or if
   it predates the stored approach data.
3. Run the clean copy. A trajectory split into parts runs part by part, each preceded by the approach above.
   `cartesian_playback` on its own refuses a split trajectory and lists the parts to run; the wrappers handle them.
4. Passing a clean ref directly (`data/preprocessed/001_…/x.csv#name`) runs it as-is if its set matches the current
   config. Otherwise it's refused rather than run under a different box.

Opt out with `--no-fit` to play the file exactly as given (still safety-checked; no approach move in
`play_cartesian_ik.sh`). `--refit` forces a new fit. Joint trajectories are never fitted.

```bash
python -m src.xarm_pipeline.safety.preprocessed list                        # sets, their configs and contents
python -m src.xarm_pipeline.safety.preprocessed which [--config path.yaml]  # set for a config, if any
python -m src.xarm_pipeline.safety.preprocessed resolve "x.csv#name"        # clean ref(s) to run (fits if needed)
```

Missing orientation is filled from `fit.default_orientation_rpy_deg`. `--default-orientation-rpy` only applies with
`--no-fit`. `play_cartesian_ik_record.sh` exits with the playback's exit code (2 = rejected, 3 = safety stop, 124 =
recording time ran out first). Pass extra recorder flags with `XARM_RECORD_ARGS="--headless --arm-only --walls"`.

## Measuring the cell and checking the gripper model on the real robot

`scripts/control/measure.sh` (set `XARM_IP` to the robot) works entirely in the **robot base frame**. It computes
positions from the joint angles with the same arm and gripper model the safety checks use, so what it reports is
exactly what the checks will see.

### 1. Give the walls by touching them (`walls`)

1. Jog the arm (Studio, slowly) or hand-guide it until a chosen part of the gripper **just touches** a wall.
   Ideally use the finger tips (`tip`, the `tool_offset_mm` point): a point contact is the most precise.
2. Type the wall name, plus the part if it isn't the tip, and press Enter: `x_max`, `y_min box`, `z_min tip`, ...
   Walls: `x_min/x_max` (behind/in front), `y_min/y_max` (robot's right/left), `z_min/z_max` (floor/ceiling).
   Parts: `tip`, `box` (the attachment), `tool` (round gripper model), `flange`.
3. Touch each wall **twice, far apart** (for example 300 mm apart along the wall). The tool averages the touches and
   warns if a wall is tilted more than 1° from the robot's axes. In that case use the innermost touch, or rotate the
   base yaw in the box form.
4. `done` prints a `workcell: walls_in_base_mm:` block (also saved under `outputs/measurements/`). Paste it into
   `config/safety.yaml`, replacing both `workcell:` and `robot_base_in_world:`.

No arithmetic is needed: for "the box touched y_max", the wall coordinate is the box's outermost y at that pose.
The tool computes that from the joints, including the box's current orientation.

### 2. Check the gripper model, without moving (`where`)

`measure.sh where` prints the flange and tip positions, the attachment's extent along each base axis (and its
corners), the round tool model's extent, and the clearance to each wall with the closest part. Put the arm in a pose
where you can measure with a ruler, for example tool pointing down with J7 at 0° and then at 90°. Check that:
* the tip height above the table matches `tool tip` z (minus the table height in the base frame);
* the attachment's lowest edge and its two side faces match the printed `attachment box occupies` ranges. At J7 = 90°
  the long (y) dimension should swap to the other horizontal axis;
* `python -m src.xarm_pipeline.sim_scene_overlay --url http://<robot ip>:18333` opens Studio with the walls and the
  yellow attachment drawn on the live arm. They should line up with what you see.

`measure.sh watch` prints the live minimum clearance (and the attachment's own) while you jog. Approach a wall slowly
and compare the printed clearance with a ruler. `--stop` sends a stop when inside the margin.

### 3. A real stop test, against a virtual wall in free space (`wall-test`)

`measure.sh wall-test --direction y+` (or `x+ x- y- z+ z-`):
1. Finds the gripper's outermost part along the direction (for `y+` with the tool down, that's the attachment) and
   puts a **virtual wall** `--gap` mm (default 80) beyond it. All real walls stay active. It refuses if the virtual
   wall would be beyond a real one.
2. Prints the plan and asks for `yes`: a straight move of gap + 40 mm at 20 mm/s, with preprocessing bypassed so
   that only the runtime guard can stop it.
3. Runs it. The guard should stop the arm with that part `safety_margin_mm` (about 20 mm) short of the virtual wall.
   It reports the flange travel and the model gap, then recovers and returns to the start pose.

**Physical check:** before running, put a marker (ruler edge, card) `--gap` mm beyond the real attachment's outer
edge. After the stop, the real gap between the edge and the marker should equal the printed model gap, within a few
mm. A smaller real gap means the attachment is bigger than configured. For `y+` the printout also says how much later
the stop would have come without the attachment (40 mm with 180 vs 100 mm widths). Seeing the early stop proves the
box is being used.

Simulator results (2026-10-14, current config): `y+` stopped on `tool_box` at 20.6 mm after 59.4 mm of travel
(predicted 50–60). `x+` stopped on `wrist` at 20.0 mm. `z-` (gap 30) stopped on `tool` at 20.4 mm. All returned to
the start pose.

Suggested order on the real arm: `where` (no motion) → live overlay → `wall-test` with the default slow speed,
`y+` and `y-` first (the attachment is the outermost part there), hand on the e-stop. Then measure the walls
with `walls`.

## Self-collision of the gripper with the robot's own body

The wall checks only know the walls. Two sim runs were stopped by the firmware's own self-collision check
(controller error **C22**, active because the sim's end-tool model was set to the xArm Gripper):
* a straight path from the ready pose to a corner behind the robot swung the hanging gripper through the base column;
* a fitted recorded trajectory started in a folded wrist pose (J4 119°, J6 180°) with the gripper against the
  forearm.

So every layer now also checks the end effector against the robot's body (`Workcell.body_slack`):
* the wrist links, the tool capsule and the gripper box against the **base column** (floor → shoulder, `radii_mm.base`);
* the tool capsule and the gripper box against the **upper arm, elbow, elbow offset and forearm**. The gripper's
  mounting end (< 40 mm from the flange) is excluded, because it sits next to the wrist housing in every pose.

Clearance must be at least `safety_margin_mm`. The pre-run check rejects violations (this can't be clipped), the
runtime monitor stops on the clearance minus closing speed × lookahead, and the fit and the test generator avoid
such poses. Self-collisions between other arm links are still not modelled. **Keep the firmware's self-collision
detection on, with the right end-tool model, on the real robot**: it caught both cases above.

Firmware stops are now reported by their real meaning (e.g. "C22 Self-Collision Error", "C35 Safety Boundary
Limit"). Previously every stop was labelled C35.

## Real-robot test trajectories (`safety.test_trajectories`)

Generated for the current config, into its preprocessed set (`data/preprocessed/NNN_*/tests/`), run as-is:

```bash
python -m src.xarm_pipeline.safety.test_trajectories corners    # slow (50 mm/s) moves into each cell corner + corner_tour
python -m src.xarm_pipeline.safety.test_trajectories dynamic    # strokes along x/y/z at 20/50/100 % of the limits
scripts/control/play_cartesian_ik_record.sh "data/preprocessed/NNN_*/tests/workspace_corners.csv#corner_tour" servo 300
```

* **corners:** the gripper **tip**, the centre of the gripper box's far face, is brought as close to each corner
  as every check allows: walls + margin + 5 mm, robot body, singularity margin (σ ≥ 25 mm/rad). The tool
  orientation is free. Each corner tries tool axes from straight down to aimed at the corner, with two spins
  about the tool axis, and keeps the best. The upper corners end up tilted about 75–147°, pointing up and out.
  `corner_tour` then visits the corners **directly corner to corner**: around the bottom ring, then back around
  the top ring, choosing the best of several visiting orders. Each leg is a straight tip line whose arm posture
  morphs between the two corners' searched postures (IK biased in the null space,
  `kinematics.ik_step_biased`). Where that fails, the leg retreats toward the middle of the cell, turns there, and
  approaches the next corner. Only if both fail does it go back along the search path through the ready posture
  (currently 2 of 9 legs; the first and last legs naturally start and end at the ready pose).
  The tour is a **joint trajectory**, so the robot executes exactly the checked joints. A Cartesian leg would leave
  the 7th degree of freedom to the firmware IK, which can end up in a different posture than the one checked.
  Play it with `python -m src.xarm_pipeline.playback --trajectory "<set>/tests/workspace_corners.csv#corner_tour" --control servo`.
  The new `servo` joint mode streams `set_servo_angle_j` at the CSV timestamps, after a joint move to the first
  sample; `position` mode still queues stop-and-go moves at 20°/s.
* **dynamic:** rest-to-rest strokes (x 200 mm, y/z 250 mm) with sine-shaped acceleration. "100 %" = the largest
  (v, a) along that stroke within tool 1000 mm/s and 50 m/s², joints 180°/s and 1146°/s² (firmware limits), computed
  through IK. In practice the joint acceleration limit keeps the tool to about 3–8 m/s². Each stroke's centre is
  chosen so the runtime monitor's speed-dependent stop is not predicted to fire, with 25 mm spare for tracking lag.

## Recorded trajectories (`safety.recorded`)

```bash
scripts/control/prepare_recorded.sh data/recorded_trajectories/5ec359d2 2.5 18 9 19   # time scale 2.5 = 2.5x slower
```

Positions only; the recorded orientation is dropped and the gripper points down. The steps:
1. time-scale;
2. cut spans over the tool speed/acceleration limits (90 % of the robot's);
3. fit or split into the cell;
4. check joint speed and acceleration through IK, then cut and refit once if needed.

Every cut is reported as a NOTICE with its time spans, plus the time scale that would have needed no speed cuts.
Parts cut out of a movement are **eased** in and out of rest along the same path, and the output is **densified to
100 Hz** with a cubic curve through every recorded sample. In servo mode, setpoints 42 ms apart (60 Hz slowed
2.5×) made the arm dash and wait at twice the planned speed, and the monitor stopped it.

Sim results for the 5ec359d2 slate at 2.5×: 0018, 0009 and 0019 are kept 100 % (0018 in 3 parts, 0019 in 2). At
1.5×, 0018 loses 8 % to speed cuts. The estimated slowdown needed for no speed cuts is about 1.9–2.4×. All parts of
0018 at 2.5× ran in the sim without a stop: peak tracking error 7–12 mm, closest approach 52 mm.

**Runtime throughput.** The per-command check reuses the joints the pre-run check validated for each sample,
instead of solving IK again. The soft-limit check reads the monitor's joints instead of making a network round
trip. At 100 Hz this took command lag from 77 ms to about 1 ms median, and peak tracking error at 1 m/s from
102 mm to 12–17 mm.

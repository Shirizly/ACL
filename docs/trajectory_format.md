# Trajectory format spec (export contract)

This is the contract for producing trajectories elsewhere (for example in a separate,
private repo) and executing them here on the xArm 7. If a file follows this spec and
passes the validator (section 6), the playback scripts accept it as-is.

There are exactly **two accepted formats**:

| Format | When to use | Played by |
|---|---|---|
| **A. Joint-space CSV** | You already have xArm 7 joint angles (planner, IK done elsewhere, recorded from this arm) | `scripts/control/play_position.sh`, `play_velocity.sh` (`src/xarm_pipeline/playback.py`) |
| **B. Cartesian (TCP pose) CSV** | You have end-effector / gripper poses (demonstrations, task-space plans). The robot firmware does the IK. | `scripts/control/play_cartesian_ik.sh`, `play_cartesian_ik_record.sh` (`src/xarm_pipeline/cartesian_playback.py`) |

If your data is in some other form (quaternions, meters, a world/camera frame, ROS
messages, another robot), convert it to A or B first. Section 5 has recipes for the usual cases.

> **Recommendation today:** use **format B in `servo` mode** for anything where timing
> matters. It is the only path that streams setpoints at the CSV's timestamps. Joint
> *position* playback does not follow the CSV timing (see A.4).

---

## 1. File rules (both formats)

| Rule | Detail |
|---|---|
| File type | Plain-text CSV, extension `.csv`. One trajectory per file, or several grouped by a `traj` column (section 3). |
| Encoding | UTF-8 (a UTF-8 BOM is tolerated) |
| Delimiter | Comma `,` only. Semicolon or tab files are rejected. |
| Decimal separator | `.` only. No thousands separators. Scientific notation (`1e-3`) is fine. |
| Header | First line, required. Column names are **lowercase, exact, case-sensitive, no surrounding spaces** (`x`, not `X` or ` x`). |
| Column order | Any order. Columns are matched by name. |
| Extra columns | Ignored by playback, including a pandas index column (`Unnamed: 0`). **They are still in the file**, so drop anything private before exporting. |
| Rows | One row = one setpoint. At least 2 rows. |
| Empty cells / NaN / inf | **Not allowed** in any column you include. The only exceptions are leaving the orientation columns out entirely (B.3), and, in a grouped file, a column that is blank for *every* row of a trajectory (section 3). The loader does *not* catch NaN in `x,y,z` or `q*`; the validator does. |
| Numbers | Plain floats. No units in cells, no quotes needed. |

## 2. The time column `t` (both formats)

- `t` is **required**, in **seconds**, float.
- It must be **strictly increasing**. It **should start at `0.0`**.
- Each row is a setpoint to be sent at time `t` after playback start. Rows are not
  interpolated on the host: what you give is exactly what gets sent, one command per row.
- Sampling may be uniform or variable. Recommended `dt` is **0.01–0.03 s** (33–100 Hz).
  - Below about 4 ms the Python send loop cannot keep up.
  - Above about 100 ms, servo mode gets large jumps between setpoints, which causes jerky motion or a firmware overspeed error.
- How `t` is used differs per mode. Read the mode notes in A.4 and B.5.

## 3. Several trajectories in one file (optional)

Add a column named **`traj`** holding each row's trajectory name. All rows with the same name
form one trajectory. Everything else in this spec applies to each trajectory on its own.

```csv
traj,t,x,y,z,roll,pitch,yaw
pick_left,0.00,300.0,100.0,250.0,3.141593,0.0,0.0
pick_left,0.02,300.5,100.0,250.0,3.141593,0.0,0.0
place_bin,0.00,250.0,-150.0,200.0,3.141593,0.0,0.0
place_bin,0.02,250.0,-150.5,200.0,3.141593,0.0,0.0
```

| Rule | Detail |
|---|---|
| Names | Any text without `#`. They are read as text, so `01` stays `01`. Avoid names that look like integers; see selection below. No blank `traj` cells. |
| Time | `t` restarts at `0.0` for each trajectory and is strictly increasing within it. |
| Row order | Keep each trajectory's rows together, in time order. Interleaving works but the validator warns. |
| Columns | The union of what the trajectories need. A column that is **blank for every row** of a trajectory counts as absent for that trajectory. For example, one trajectory can have velocity columns and another none, or one can have orientation and another not. A column that is **partly** blank within a trajectory is still an error. Joint and Cartesian trajectories may share a file this way. |
| Without `traj` | The file is one trajectory, exactly as before. |

**Selecting one.** Wherever a trajectory path is accepted (every playback script and CLI, the
validators, the safety check), append `#name` or `#index`:

```bash
scripts/control/play_cartesian_ik.sh "data/trajectories/bank.csv#pick_left" servo
scripts/control/play_cartesian_ik_record.sh "data/trajectories/bank.csv#2" servo 30
```

- An exact name match wins. Otherwise an integer is the **0-based** position in order of first
  appearance.
- Quote the argument in the shell. `#` is safe inside quotes and inside a word, but a habit of
  quoting avoids surprises.
- A grouped file with more than one trajectory and no selector is refused, and the error lists what's inside.
  A grouped file with exactly one trajectory needs no selector.
- To list the contents:
  `python -m src.xarm_pipeline.trajectory_io data/trajectories/bank.csv` prints
  `[index] name: rows, t range -> ready-to-use path#name`.

---

## Format A: joint-space CSV

### A.1 Columns

| Column | Required | Unit | Meaning |
|---|---|---|---|
| `t` | yes | s | time from start |
| `q1` … `q7` | yes | **rad** | joint positions, `q1` = base joint (J1) … `q7` = wrist/flange joint (J7) |
| `dq1` … `dq7` | only for velocity playback | **rad/s** | joint velocities. All seven or none. A partial set is ignored. |

```csv
t,q1,q2,q3,q4,q5,q6,q7
0.00,0.0,0.0,0.0,0.5,0.0,0.5,0.0
0.02,0.001,0.0005,0.0,0.501,0.0,0.5,0.0
```

### A.2 Joint convention

Angles must be in the **xArm firmware's own joint convention**: the same zero position
and sign as `arm.get_servo_angle(is_radian=True)` returns, and as UFACTORY Studio shows (in degrees).
All-zeros is the folded "zero" pose: with zero TCP offset the flange is at (206, 0, 120.5) mm, pointing
straight down (roll=π, pitch=0, yaw=0). This was verified with the simulator's firmware FK. If your source uses a different model (URDF from
another package, DH with offsets, another robot), check that zero and sign match by
commanding a known pose in the simulator. Don't assume they match.

### A.3 Joint limits (xArm 7, rad, from the SDK's `x_config.py`)

| Joint | Limit | Note |
|---|---|---|
| q1, q3, q5, q7 | [-2π, 2π] | |
| q2 | [-2.0595, 2.0944] | |
| q4 | **[-3.9270, 0.1920]** on hardware revision X3/X13, **[-0.1920, 3.9270]** on X4 | The sign depends on the arm's revision. Confirm which one the real arm is before exporting. |
| q6 | [-1.6930, π] | |

Peak joint speed must stay ≤ π rad/s (180°/s, the hardware maximum). Keep it much lower
in practice.

### A.4 How joint trajectories are executed (current behavior)

- **`--control position`** (`play_position.sh`): mode 0, one `set_servo_angle(q_row, wait=False)` per row at time `t`.
  Mode 0 **queues** planned point-to-point moves at the SDK default joint speed (**20°/s**).
  So the robot works through your rows at its own pace, and **`t` is effectively ignored**.
  The path is followed, but not the timing. The first row is approached the same way from
  wherever the arm is.
- **`--control velocity`** (`play_velocity.sh`): mode 4. At each `t`, row `dq` is sent
  with `vc_set_joint_velocity` (synchronized joints) and held until the next row. After the
  last row, zero velocity is sent. **`q` columns are required but not used.** Motion starts
  from wherever the arm currently is, so you must move the arm to `q[0]` first yourself.
  Drift between the integrated `dq` and your `q` is not corrected.
- Neither joint mode runs the orientation clamp or the joint soft-limit check that Cartesian playback has.
  Both *do* run the workcell / safety-wall checks (B.5 step 4, `docs/safety_walls.md`). The first row must
  therefore already be inside the walls. For example, the all-zeros pose puts the default 172 mm gripper
  below the floor.

---

## Format B: Cartesian (TCP pose) CSV

### B.1 Columns

| Column | Required | Unit | Meaning |
|---|---|---|---|
| `t` | yes | s | time from start |
| `x`, `y`, `z` | yes | **mm** | TCP position in the **robot base frame** |
| `roll`, `pitch`, `yaw` | recommended (see B.3) | **rad** | TCP orientation, xArm RPY convention (B.2) |
| `gripper_roll`, `gripper_pitch`, `gripper_yaw` | alternative | rad | **Exact alias** for `roll,pitch,yaw`. Used only if `roll,pitch,yaw` are absent. There is no gripper actuation here, and these do not mean "relative to the gripper". |
| `vx`, `vy`, `vz` | only for `velocity` mode | mm/s | TCP linear velocity, base frame |
| `vroll`, `vpitch`, `vyaw` | only for `velocity` mode | rad/s | TCP angular velocity sent to `vc_set_cartesian_velocity` as `[rx, ry, rz]` |

All six velocity columns or none (a partial set is ignored). `x,y,z` are required even in velocity mode.

```csv
t,x,y,z,roll,pitch,yaw
0.00,300.0,0.0,250.0,3.141592653589793,0.0,0.0
0.02,300.5,0.2,250.1,3.141592653589793,0.0,0.0
```

### B.2 Frames and orientation, exactly

**Base frame.** Origin is at the center of the robot's base mounting surface. `+z` points
up, perpendicular to the mounting surface. `x`/`y` are fixed to the base casting: positive
`x` is the direction the flange sits in at the all-zeros pose (x = 206 mm there). This is the frame of
`arm.get_position()` and of the Studio UI. The J2 (shoulder) axis is 267 mm above the
origin, and the reach from there is about 700 mm.

**Which point `x,y,z` is.** It is the **TCP**: the flange center plus the TCP offset
configured in the firmware (`set_tcp_offset`, or Studio → Settings → TCP). With zero
offset it is the center of the J7 flange face. If your data describes a gripper tip,
fingertip, or tool point, do one of the following:
1. Set the same offset on the robot (Studio, or `arm.set_tcp_offset([0,0,L,0,0,0])` with `L` in mm along the flange z), **or**
2. Convert your poses to the flange before export: `p_flange = p_tip − R · [0, 0, L]`, where `R` is the tip orientation.

Whichever you choose, write it down next to the data. A mismatch shifts every point by `L`
along the tool axis, and nothing will warn you.

**Orientation convention.** `roll, pitch, yaw` are fixed-axis (extrinsic) rotations about base
**X, then Y, then Z**:

```
R = Rz(yaw) · Ry(pitch) · Rx(roll)
```

The columns of `R` are the TCP's x/y/z axes expressed in the base frame. In scipy this is
`Rotation.from_euler("xyz", [roll, pitch, yaw])`, with **lowercase** `"xyz"` (uppercase
`"XYZ"` is intrinsic and gives a different result). This was verified against the firmware's own FK:
rotating J7 by Δ changes the reported RPY exactly as `R·Rz(Δ)` under this convention, and the intrinsic
variant is off by up to 0.95. Reference orientations:

| roll, pitch, yaw (rad) | Tool z-axis (approach direction) |
|---|---|
| π, 0, 0 | straight down (−z of base), the usual "home" orientation |
| 0, π, 0 | also straight down, but tool x flipped (same as π,0,π) |
| π, 0, θ | down, rotated by θ about base z |

- Any branch is accepted (−π…π or unwrapped), but **consecutive rows must be continuous**.
  Playback unwraps each angle to within ±π of the previous one.
- Avoid `|pitch|` near π/2 (gimbal lock). There, roll and yaw become ill-defined and small
  pose changes produce large RPY jumps.

### B.3 Missing orientation

If **none** of `roll,pitch,yaw` / `gripper_*` are present, every row gets the fallback orientation:
`--default-orientation-rpy "r,p,y"` (rad) if given, otherwise the robot's orientation at
playback start. A partial group (for example only `yaw`) is an error. Per-row blanks are not allowed.

### B.4 Velocity columns (only for `velocity` mode)

Each row's `[vx, vy, vz, vroll, vpitch, vyaw]` is sent at time `t` with
`vc_set_cartesian_velocity(..., is_tool_coord=False)` and held until the next row. Zero is sent
after the last row. These values are not derived from the pose columns, so keep them
consistent yourself. Note that the angular terms are the firmware's rotational speeds about
the base axes, **not** `d(roll)/dt` etc. Verify in the sim before relying on angular velocity playback.

### B.5 How Cartesian trajectories are executed

Modes (`play_cartesian_ik.sh <csv> <mode>`):

| Mode | Robot mode | Call per row | Timing honored? |
|---|---|---|---|
| `servo` (default) | 1 | `set_servo_cartesian(pose)`: streaming, firmware IK, the latest setpoint wins | **yes** |
| `pose` | 0 | `set_position(..., wait=False)`: queued linear moves at the default 100 mm/s | no (queued) |
| `velocity` | 5 | `vc_set_cartesian_velocity(twist)` | yes |

Processing applied to your file before sending (`cartesian_playback.py`):

1. **Orientation fill** (B.3), then **orientation continuity**: each angle is unwrapped
   near the previous row, starting from the robot's *current* orientation. Each per-row change is
   **clamped to `--max-orientation-step-deg` (default 8°)**. If your orientation changes faster
   than 8° per row, the commanded orientation silently lags. Sample densely enough.
2. **Timeline re-basing**: row 0 is sent at 0 s and held for `--start-hold-seconds` (0.5 s).
   Row `i ≥ 1` is sent at `hold + (t[i] − t[1])`. Only the differences between your `t` values matter.
3. **Joint soft-limit check** before every command (`--joint-limit-deg`, default 365°). Playback aborts if it is exceeded.
4. **Workcell / safety-wall checks** (on by default): the whole trajectory is checked against the
   workcell box before anything moves, and is either rejected (exit code 2) or, with `--safety-policy clip`,
   shifted back inside. During execution, every command is checked and a monitor stops the arm (exit code 3).
   This applies to joint playback too. See `docs/safety_walls.md`.

The robot does **not** automatically travel to row 0 in `play_cartesian_ik.sh`. Servo mode would
try to jump there, so start near row 0. `play_cartesian_ik_record.sh` does this for you: it
first runs `prepare_cartesian_start.sh`, a linear `set_position` move to row 0. That approach
move is a straight line from wherever the arm is.

Keep per-row TCP steps small for servo mode. The bundled samples use about 0.6–2.8 mm per row
(20–55 mm/s). The validator errors above 500 mm/s by default. Treat anything above about 250 mm/s as
something to try in the sim first.

---

## 4. Which format should I export?

- Your data is **end-effector poses** (teleop, mocap, demonstrations, task-space planner): export **B**.
- Your data is **xArm 7 joint angles** with meaningful timing: export **A with `dq` columns** and
  use velocity playback, or export **B** by running FK first. Joint position playback will not
  respect your timing (A.4).
- Your data is from **another robot**: retarget to TCP poses and export **B**.

## 5. Conversion recipes

All snippets assume numpy/pandas/scipy in the exporting repo. This repo does not need scipy.

```python
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot

# Meters -> millimeters
xyz_mm = xyz_m * 1000.0

# Quaternions -> xArm RPY. CHECK YOUR SCALAR ORDER: scipy wants [qx, qy, qz, qw] (scalar last).
rpy = Rot.from_quat(quat_xyzw).as_euler("xyz")          # lowercase "xyz" = xArm convention
rpy = np.unwrap(rpy, axis=0)                             # keep rows continuous

# Rotation matrices (N,3,3) -> RPY
rpy = np.unwrap(Rot.from_matrix(R).as_euler("xyz"), axis=0)

# Degrees -> radians
rpy = np.radians(rpy_deg)

# Poses in a world/table frame -> robot base frame
# T_world_base = pose of the robot base in your world frame (4x4). For the workcell defined in
# config/ for the safety walls, that is (x, y, z, yaw) of the base relative to the floor center.
T_base_world = np.linalg.inv(T_world_base)
p_base = (T_base_world[:3, :3] @ p_world.T).T + T_base_world[:3, 3]
R_base = T_base_world[:3, :3] @ R_world                  # (N,3,3) via np.einsum or a loop

# Tool tip -> flange (if you don't set a TCP offset on the robot), L = tool length along flange z, mm
p_flange = p_tip - Rot.from_euler("xyz", rpy).apply([0.0, 0.0, L])

# Resample variable-rate data to a fixed dt (positions only; re-derive orientation with Slerp)
t_new = np.arange(0.0, t[-1], 0.02)
xyz_new = np.column_stack([np.interp(t_new, t, xyz[:, k]) for k in range(3)])
from scipy.spatial.transform import Slerp
rpy_new = np.unwrap(Slerp(t, Rot.from_euler("xyz", rpy))(t_new).as_euler("xyz"), axis=0)
```

### Reference exporter (copy into the other repo)

```python
import numpy as np, pandas as pd

def export_cartesian_csv(path, t, xyz_mm, rpy_rad):
    """t: (N,) s; xyz_mm: (N,3) TCP in xArm base frame, mm; rpy_rad: (N,3) xArm RPY (extrinsic xyz), rad."""
    t = np.asarray(t, float); xyz = np.asarray(xyz_mm, float); rpy = np.unwrap(np.asarray(rpy_rad, float), axis=0)
    assert t.ndim == 1 and xyz.shape == (len(t), 3) and rpy.shape == (len(t), 3) and len(t) >= 2
    assert np.isfinite(t).all() and np.isfinite(xyz).all() and np.isfinite(rpy).all()
    assert (np.diff(t) > 0).all(), "t must be strictly increasing"
    assert np.abs(xyz).max() > 2.0, "looks like meters; expected mm"
    t = t - t[0]
    df = pd.DataFrame({"t": t, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
                       "roll": rpy[:, 0], "pitch": rpy[:, 1], "yaw": rpy[:, 2]})
    if path is None:
        return df
    df.to_csv(path, index=False, float_format="%.6f")

def export_bank_csv(path, frames: dict):
    """Group several trajectories into one file. frames: {name: DataFrame from the functions above
    (built in memory, i.e. call them with path=None)}."""
    for name in frames:
        assert "#" not in name and name.strip() == name and name, f"bad trajectory name {name!r}"
    df = pd.concat([f.assign(traj=n) for n, f in frames.items()], ignore_index=True)
    df = df[["traj"] + [c for c in df.columns if c != "traj"]]
    df.to_csv(path, index=False, float_format="%.6f")   # blanks where a trajectory lacks a column

def export_joint_csv(path, t, q_rad, dq_rad_s=None):
    """q_rad: (N,7) xArm 7 joint angles in firmware convention, rad. dq optional (N,7) rad/s."""
    t = np.asarray(t, float); q = np.asarray(q_rad, float)
    assert q.shape == (len(t), 7) and np.isfinite(q).all() and (np.diff(t) > 0).all()
    df = pd.DataFrame(q, columns=[f"q{i}" for i in range(1, 8)])
    df.insert(0, "t", t - t[0])
    if dq_rad_s is not None:
        dq = np.asarray(dq_rad_s, float); assert dq.shape == q.shape and np.isfinite(dq).all()
        df[[f"dq{i}" for i in range(1, 8)]] = dq
    if path is None:
        return df
    df.to_csv(path, index=False, float_format="%.6f")
```

## 6. Validating before you run

Format check (no robot needed). Exit code 0 = OK, 1 = errors, 2 = unreadable:

```bash
.venv/bin/python -m src.xarm_pipeline.validate_trajectory path/to/*.csv            # human-readable
.venv/bin/python -m src.xarm_pipeline.validate_trajectory traj.csv --robot-type x4 --json
.venv/bin/python -m src.xarm_pipeline.validate_trajectory bank.csv           # checks every trajectory in a grouped file
.venv/bin/python -m src.xarm_pipeline.validate_trajectory "bank.csv#pick_left"   # or just one
```

It checks header names (including case and whitespace mistakes), delimiter, NaN/inf, strictly
increasing `t`, sample-rate sanity, unit sanity (meters vs mm, degrees vs rad), the xArm 7 TCP box
and reach, joint limits (with the J4 revision ambiguity), peak joint and TCP speed, `dq`-vs-`q`
consistency, orientation step size against the playback clamp, and gimbal-lock proximity.

Then run the workcell / safety-wall check. It needs the safety config (`config/safety.yaml`) to describe the
real cell and tool. For Cartesian files the result depends on the arm's start joints, so point it at the
robot (or sim) with `XARM_IP`:

```bash
scripts/control/validate_trajectory_safety.sh traj.csv reject    # exit 0 = ok, 2 = rejected, 3 = config error
scripts/control/validate_trajectory_safety.sh traj.csv clip      # also writes a sanitized CSV + JSON report
```

The same check runs automatically at the start of every playback (see `docs/safety_walls.md`).
Cartesian playback assumes only relative motion matters. By default it runs a **clean copy**: the trajectory
re-placed, rotated, or split to fit the current workcell, stored under `data/preprocessed/NNN_*/` (one set per safety
setup, created on first use). `--no-fit` plays the file as given. See "Fitting trajectories into the cell" and
"Preprocessed sets" in `docs/safety_walls.md`. Then do a dry run
in the simulator (`play_cartesian_ik_record.sh <csv> servo <seconds>` gives a video plus a tracking-error report).

## 7. Exporter checklist

- [ ] Comma-separated CSV, header row, lowercase exact column names; one trajectory per file, or grouped with a `traj` column (names without `#`, `t` restarting at 0 per trajectory)
- [ ] `t` in seconds, starts at 0, strictly increasing, dt ≈ 10–30 ms
- [ ] Format A: `q1..q7` in rad, xArm firmware joint convention, J4 sign matches the arm's revision; `dq1..dq7` in rad/s if using velocity playback
- [ ] Format B: `x,y,z` in **mm** in the **robot base frame**, of the agreed **TCP point**; `roll,pitch,yaw` in **rad**, **extrinsic X→Y→Z** (`R = Rz·Ry·Rx`), continuous between rows
- [ ] No NaN/inf/blank cells; no private extra columns
- [ ] Orientation changes < 8° per row; no pitch near ±90°
- [ ] Recorded alongside the data: the TCP offset assumed, the source frame → base transform used
- [ ] Passes `validate_trajectory` and the safety-wall check, and is dry-run in the sim

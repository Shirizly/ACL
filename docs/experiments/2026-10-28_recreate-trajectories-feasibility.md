# Can the xArm 7 recreate the recorded sensor trajectories (arm alone, extra motor, stick)?
Date: 2026-10-28 · Status: complete (single pass of a heuristic placement search; see caveats) · Code: `src/xarm_pipeline/analysis/recreate_feasibility.py` · Results: `docs/experiments/data/2026-10-28_recreate-trajectories-feasibility.csv` (one row per design × trajectory); raw per-job JSON in `outputs/feasibility/` (git-ignored, local only)

## Question
The 39 recorded trajectories (`data/recorded_trajectories/5ec359d2`, 60 Hz mocap, position + orientation of a hand-held sensor) need recreating with the robot. Three options:
1. **Arm alone**: the sensor is rigidly mounted on the gripper and the arm tracks the full pose.
2. **Arm + one motor**: the sensor sits on the shaft of an extra motor mounted on the gripper at an angle. The motor supplies part of the rotation.
3. **Stick**: the gripper holds a stick and only the stick tip's *position* is recreated (orientation ignored). Can it extend the range so whole trajectories run in one motion? Per the user, the stick tests use **no workcell walls**; only self-collision of the stick/gripper with the robot counts.

## Data facts (survey of all 39)
- Duration 4–15 s. Path bounding box: median 549 × 357 × 175 mm, largest 703 mm (0031) and 947 mm (0019, along y).
- Speed: peak 1.5–2.3 m/s. The xArm's TCP limit is 1 m/s, so even a perfect arm needs ≥ ~2× slowdown (≥ 2.1× at 90% of the limits, as used throughout).
- Orientation range is **small**: every trajectory stays within 20–37° of its own mean orientation. Angular speed peaks at 265–705 °/s (95th percentile ≈ 190 °/s).
- Angular velocity is nearly **isotropic**: RMS per principal axis is 31 / 48 / 50 °/s. No single axis carries most of the rotation.
- 0002 and 0025 hold the sensor about 180° differently from the other 37.

## What was run
For each design, the existing fitter (`safety.fit.Fitter`) placed each whole trajectory by translation + yaw. Tilting is not allowed, so orientation keeps its relation to gravity. The fitter maximized clearance using the same continuation IK, capsule wall model, self-collision (`body_slack`) and runtime-monitor prediction as playback. "Whole" means one placement with ≥ 5 mm slack for the full trajectory, with no splitting. The fit was timed at 2.5× slowdown, which sets the monitor margin. The **required slowdown** is max(peak joint speed / 162 °/s, √(peak joint acc / 1031 °/s²)) along the fitted joint path, i.e. 90% of the robot limits, with acceleration taken on a 5-sample moving average. Robot limits come from `safety.test_trajectories`. The workcell is `config/safety.yaml` as of today: walls in base frame x −490…530, y −390…460, z 20…1200 mm, 20 mm margin, standard gripper box.

| design | description |
|---|---|
| `tooldown` | reference: positions only, gripper pointing down, sensor point = gripper tip (172 mm) |
| `arm_down` | Q1: full orientation; per-trajectory mount chosen so the mean orientation = gripper pointing down |
| `arm_fwd` | Q1 variant: mean orientation = gripper horizontal (only trajectories 0009 and 0018; clearly worse, dropped) |
| `motor` | Q2: motor shaft along sensor axis a_s = (−0.964, −0.265, −0.024), the axis whose world direction varies least over all data. Shaft tilt from the gripper axis = angle of its mean direction from vertical. The arm points the shaft; the free rotation about the shaft is set to stay closest to gripper-down. Motor angle recovered exactly (residual < 1e-5°) |
| `tooldown_nowalls` | Q3 reference: as `tooldown`, no walls (floor = robot mounting plane kept) |
| `stick_L600_back800` | Q3 "wand": 600 mm stick from the flange, pointing from a pivot 800 mm closer to the robot axis (horizontally) to the tip. Reach = arm + stick |
| `stick_L500_up800` | Q3 "fanning pen": 500 mm stick pointing from a pivot 800 mm above the path centre down to the tip |
| `range_x1.5`, `range_x2.0` | Q3 headroom: the three largest paths (0002, 0009, 0019) enlarged 1.5× / 2× about their centre |

The stick is a 15 mm-radius capsule from the flange to the tip, checked against the robot base column and the upper-arm, elbow and forearm links. Gripper spin about the stick was searched (±90°); it does not move the tip. Wall-time: about 6–15 min per job on 8–11 shared cores, roughly 5 h in total. The run includes a mid-run restart and one job (arm_down 0017) that was re-run after a config-file race between workers, which has since been fixed.

## Results
Walled cell, all 39 trajectories:

| design | whole in one piece | slowdown median / p90 / max (whole ones) |
|---|---|---|
| tooldown (positions only) | 31/39 | 1.80 / 2.09 / 2.34 |
| arm_down (full orientation) | 29/39 | 2.30 / 2.88 / 3.62 |
| motor (full orientation) | 33/39 | 2.40 / 3.04 / 3.83 |
| any of the three | 36/39 (not 0009, 0012, 0019: the three longest paths) | — |

- Paired over the 25 trajectories that all three designs fit: slowdown median tooldown 1.79, arm_down 2.27, motor 2.22. Motor/arm ratio median 0.97; the motor is faster on 68% of them.
- The TCP 1 m/s limit alone needs a median 2.07× slowdown (max 2.54×), so full-orientation tracking adds little beyond that floor.
- arm_down: the gripper tilts at most 20–37° from straight down.
- Motor needs at original speed: shaft tilt 29–56° (median 49°) from the gripper axis. Total rotation range only 23–52°, so no continuous rotation is needed. Peak speed 196–396 °/s, p95 ≈ 100 °/s; divide by the slowdown.

No walls (stick question):

| design | whole | joint slowdown median (paired, 12 traj.) |
|---|---|---|
| tooldown_nowalls (no stick) | **39/39** | 2.28 (all 39: median 2.41, max 5.15) |
| stick_L600_back800 (wand) | 12/12 | 3.04 |
| stick_L500_up800 (fanning pen) | 10/12 | 2.99 |

| enlarged paths (0002, 0009, 0019) | 1.5× | 2× |
|---|---|---|
| no stick | 2/3 (fails 0019) · slowdown 6.25, 3.87 | 0/3 |
| wand 600 mm | 3/3 · 3.84, 3.08, 6.44 | 2/3 (fails 0019 = 1.13 × 1.89 m) · 5.63, 2.78 |
| wand 800 mm | — | 2/3 (fails 0019) · 6.24, 3.09 |
| fanning pen 500 mm | 1/3 | — |

## Answer
1. **Arm alone is possible** for most trajectories, contrary to the expectation. The recorded orientations span only ±20–37°, which the wrist covers easily. In the current cell 29/39 run whole with full orientation; the others fail on the *walls*, not orientation (positions-only fails on about as many, 31/39). Splitting, as `safety.recorded` already does, would handle the rest. The real limit is speed: no design runs in real time. About 2.1× slowdown is forced by the 1 m/s TCP limit anyway, and orientation tracking raises the joint-limited slowdown to a median of about 2.3× (max 3.6×). Confidence: good for feasibility; the fitter is heuristic, so ±2–3 trajectories per design is within search noise.
2. **One extra motor helps only marginally.** Angular velocity in these recordings is nearly isotropic, so one axis absorbs about a third of the rotation. Measured effect: 33 vs 29 whole (within noise) and about 3% less slowdown. If built anyway: shaft about 45–50° from the gripper axis, sensor on the shaft axis with its −x axis along the shaft. A small servo with ±30° travel and about 300 °/s is enough at real time (less when slowed down). Not worth the hardware for this data set unless the recordings change.
3. **A stick is not needed for these trajectories once the walls are gone.** The bare arm runs all 39 whole, and a stick only costs about 30% more slowdown. The stick does extend range: for paths 1.5–2× larger, a 600 mm wand pointing away from the robot runs whole paths the bare arm cannot. It fits about 1.3 m of tip reach from the robot axis and paths up to about 1.0 × 1.4 m, and needs less slowdown on large paths (3.8 vs 6.3). A 1.1 × 1.9 m path is too big even for an 800 mm wand. A wand works better than a stick fanning down from above.

## Caveats & issues
- The fitter is a heuristic local search, single pass. Some per-trajectory flips are search noise: for example, the fanning pen fails 0012 at a point where the search never left its start.
- Q1/Q2 use a per-trajectory sensor mount (mean orientation → gripper down). One fixed mount would raise the gripper tilt to about 36–54°, and 0002/0025 would need the mount turned 180°. Not tested.
- The slowdown is not optimized: the IK's redundancy resolution and the placement are chosen for clearance, not joint speed. Without walls the placement differs, which is why tooldown_nowalls shows higher slowdowns than tooldown.
- Stick self-collision samples the stick at 7 points, about 100 mm apart for a 600 mm stick, against the link capsules. Up to about 15 mm of true clearance can be missed, inside the 20 mm margin. Stick bending and the dynamic load of a long stick on J6/J7 are not modelled.
- The sim-firmware IK is not involved: this is the local IK model, whose redundancy resolution matches the firmware within about 10 mm (tolerance included).

# How to slow the recorded human trajectories for IMU accuracy tests, and what the robot cannot reproduce
Date: 2026-10-28 · Status: complete (analysis on TCP-level capability proxies; no IK check of the transformed paths, no hardware) · Code + raw outputs: `outputs/imu_test_scaling/` (git-ignored, local only; one sub-folder per analysis: `timewarp/`, `metrology/`, `stick/`, `critic/`, plus `imu_stats.py`, `BRIEF.md`) · Builds on [2026-10-28_recreate-trajectories-feasibility.md](2026-10-28_recreate-trajectories-feasibility.md)

## Question
The robot cannot replay the 39 recorded trajectories (`data/recorded_trajectories/5ec359d2`) in real time. They are meant for testing whether IMUs are accurate enough for wearables that capture these human motions. The question has four parts:
- How should the trajectories be slowed or transformed to keep what matters for the test?
- The user's proposal: keep the accelerations, cut them off at the speed limit, and replay fast events at a lower base speed. Does it work?
- Can a stick held by the gripper reproduce the fast parts?
- What is lost completely?

## What was run
Three independent analyses were run, followed by an adversarial cross-check that re-ran the key numbers. All use the 60 Hz mocap, smoothed over 5 samples and differentiated, so content above about 12 Hz is invisible.

The IMU stimulus is:
- specific force f = Rᵀ(a + g ẑ), in the sensor frame;
- body angular rate ω.

Robot capability comes from the Jacobian of `safety/kinematics.py` at 90% of the joint limits: 162 °/s and 18 rad/s².
1. **Transformation schemes** (`timewarp/`), evaluated on 0009, 0012, 0015, 0031 (dynamic) and 0013, 0019 (long):
   - uniform scaling (U);
   - the literal proposal: integrate the original a, clip |v| (L);
   - optimal Galilean washout (O): subtract the smoothest velocity offset c(t) with |v − c| ≤ v_max under a displacement budget;
   - tilt coordination (T);
   - variable time warp (V), and washout plus warp (OV);
   - exact replay window by window with a constant velocity offset (G).

   The feasibility proxy is |v| ≤ 0.9 m/s, |a| ≤ a_cap and |ω| ≤ 160 °/s.
2. **Metrology** (`metrology/`): which signal properties determine IMU error, and how a 2.3× slowdown changes them. This includes a Mahony orientation-filter simulation with ideal sensors at 200 Hz.
3. **Stick mechanics** (`stick/`):
   - tip speed and acceleration vs stick length;
   - instantaneous screw-axis analysis of the human motion;
   - IK replays with min-norm redundancy (crude placement, no walls);
   - stick bending, wrist load and safety.
4. **Cross-check** (`critic/`):
   - recomputed the capability of the arm with orientation held fixed;
   - ran the same Mahony filter on the stimulus of every scheme, which is the deciding test.

## Results
Human data at real time (all 39, median):
- Dynamic acceleration is above 0.5 g 10.7% of the time and above 1 g 5.6%.
- ω is above 100 °/s 9.6% of the time.
- High ω and a > 0.5 g occur together 7.7% of the time.
- In the dynamic trajectories, 62–71% of the samples with |a| > 1 g are also above 160 °/s.
- Peaks are 1.4–3.3 g and 250–615 °/s.

Arm capability with the orientation held, gripper only:

| quantity | worst direction | median direction |
|---|---|---|
| translation speed | 0.56–0.64 m/s | 0.84–0.97 m/s |
| translational acceleration from rest | 0.36–0.42 g | 0.54–0.63 g |
| angular rate | 73–139 °/s | 139–212 °/s |

- The joint-speed limits alone already cap translation below the 1 m/s TCP limit, in every direction.
- With the orientation free, the acceleration reaches 1.1–1.8 g in the median direction. A 500 mm stick raises that to about 2.6 g, but the worst direction stays at about 0.5 g.

Orientation-filter difficulty kept by each scheme: Mahony tilt RMS relative to real time, median over the 4 dynamic trajectories. Real time is 5.2° RMS.

| caps (v, a, ω) | O 0.4 m* | OV 0.4 m | OV 0.6 m | V | U | G |
|---|---|---|---|---|---|---|
| 0.9 m/s, 1 g, 160 °/s | 0.66 | 0.48 | 0.55 | 0.62 | 0.44 (s 2.3) | 0.34 |
| 0.9 m/s, 0.5 g, 160 °/s | 0.66 | 0.45 | 0.51 | 0.59 | 0.40 (s 2.5) | 0.25 |
| 0.6 m/s, 0.5 g, 160 °/s | 0.61 | 0.53 | 0.47 | 0.46 | 0.27 (s 3.5) | 0.25 |
| same, no ω limit (gimbal) | 0.61 | 0.55 | 0.49 | 0.46 | 0.27 | 0.25 |

\* O alone is infeasible: it bounds velocity only, and its acceleration peaks at 2.3–2.7 g.

Other per-scheme results:
- **L:** the velocity is clipped 27–31% of the time. The robot moves *backwards* while the human decelerates, and the position random-walks by 1.2–5 m.
- **OV:**
  - signals exact in about 77% of samples, so ω and the spectra are exact there;
  - duration ×1.1;
  - samples above 0.5 g kept: 0.15–0.17 of samples (real time: 0.25);
  - specific-force error 0.48–0.55 g RMS (U: 0.59–0.62 g).
- **U:**
  - all rates divided by s, all accelerations divided by s², bandwidth divided by s (7 Hz → about 3 Hz), duration ×s;
  - median |a_dyn| 0.06 → 0.011 g, which is below typical accelerometer offset residuals;
  - samples above 0.5 g drop to 0.
- **T:** moves the accelerometer error into the gyro, an artefact of 28 °/s RMS and 80 °/s peak. Rejected.
- **Stick:**
  - Human fast motion is about 82% rotation, at a median instantaneous radius of 0.42 m. But the pivot moves by 0.37 m within a stroke, so the best fixed lever lowers the needed flange speed by only 12%.
  - A position-only replay with a 600 mm stick needs about 1.3× slowdown, but the sensor's ω is uncorrelated with the human's (error 228 °/s RMS), and the gravity projection is off by 0.32 g RMS.
  - Bending mode of a 0.6 m carbon tube: about 20 Hz with a 150 g tip. Gripper friction cannot hold the moment, so the stick has to be bolted to the flange.
  - The xArm ServoJ mode applies a 1.2 m/s TCP limit (UFACTORY C60 support article). A stick tip faster than the declared TCP bypasses a safety function.

## Answer
1. **The binding limit is the arm's acceleration capability, not the speed limit.** With orientation tracked the arm makes only about 0.4–0.6 g of translational acceleration. The humans exceed 1 g 5.6% of the time, with peaks of 1.4–3.3 g, and usually while rotating faster than the wrist can (more than 160 °/s). Confidence: medium. The critical number is the 20 rad/s² joint-acceleration limit, an SDK planner parameter that has never been measured in servo mode.
2. **The user's proposal does not work literally, but its well-posed form is sound.** An IMU cannot sense constant velocity, so replaying an event at a lower base speed (a Galilean shift with the orientation history kept) is exact for both the accelerometer and the gyro. Doing it smoothly is the washout (O). It removes the velocity problem for an extra 0.4–0.6 m of displacement. It cannot fix the acceleration and angular-rate caps, so the infeasible events still need local slowing (OV).
3. **No scheme keeps the orientation-filter difficulty.** Washout plus warp keeps about half; uniform scaling keeps 27–44%. Washout plus warp's real advantages over uniform scaling are elsewhere:
   - ω and spectra exact in about 77% of samples, where uniform scaling halves every rate;
   - duration ×1.1 instead of ×2.3;
   - some occupancy above 0.5 g instead of none.
4. **A slowed replay overstates IMU accuracy.** Orientation-filter tilt error is 1.7–2.4× lower at 2.3× slowdown in the 39-trajectory simulation, and the "clean gravity" fraction rises from 66% to 91%. Errors must be reported stratified by |a_dyn|/g and ω bins, then weighted by the human occupancy of each bin. A single RMS from a slowed replay must not be quoted.
5. **A stick on its own does not help IMU testing.** It cannot raise the angular rate, and the human rotation pivot moves within a stroke, so the sensor at the tip gets the wrong ω and the wrong gravity projection. A stick with a fast 2–3-axis tip gimbal might reach the cells above 1 g. Its requirements, untested:
   - gimbal range ±45° and about 450 °/s, with the IMU at the gimbal centre;
   - stick at most 0.6 m, 25–30 mm carbon, bolted to the flange;
   - a guarded cell, with the true TCP declared to the firmware.
6. **Lost completely with this arm**, to be covered by bench rigs (rate table, pendulum/drop, shaker) and human trials with mocap:
   - dynamic acceleration above about 0.6 g with the right orientation;
   - rates above about 160–210 °/s;
   - the coincidence of high rate with corrupted gravity, which is what makes orientation filtering hard;
   - content above about 12 Hz, which the 60 Hz mocap never had either;
   - strap and soft-tissue artefacts.

   The magnetometer is the opposite case: the robot adds disturbances a wearable would not see, so ignore the magnetometer on the robot.

## Addendum: testing rotation and linear acceleration separately
Question: can rotation be replayed alone (translation not tracked) and linear acceleration alone with a stick (orientation not tracked, left free; the sensor turns with the stick and forward kinematics gives its true orientation), each closer to real time? A few combined runs would then measure the cross-effects.

Scripts in `outputs/imu_test_scaling/decoupled/`:
- `rot_only.py`, all 39 trajectories: bare-arm velocity-level IK on the 3×7 angular Jacobian, min-norm with a posture null space, mean orientation set to gripper-down.
- `lin_only_stick.py`: the stick agent's position-only IK, best of 6 placements, on 15 trajectories including the 10 dynamic ones.
- `f_split.py`: splits the accelerometer signal into its gravity and linear parts.

| test | result |
|---|---|
| Accelerometer signal, varying part | gravity re-projection R^T g: RMS 0.16 g, peak 0.44 g (about 14% of the variance). Linear R^T a: RMS 0.39 g, peak 1.4–3.3 g (about 86%) |
| Rotation only, bare arm | slowdown median 2.0, range 1.8–2.65, set by angular **acceleration** (human peaks 4,800–9,800 °/s² vs 1,031 °/s² per joint, J5 binding). Joint-rate term alone: 1.06. Joint-limit margin ≥ 27° |
| Linear only, 600 mm stick, orientation free | slowdown median 1.09, range 0.82–1.56; **7/14 run in real time**; 0019 found no valid placement. With a 300 mm stick: median 1.37 |
| Rotation of the 600 mm stick, i.e. what the IMU sees in the linear test (an optional tip gimbal would have to cancel this to hold a fixed world orientation) | up to 36–61° from its mean, 120–212 °/s peak, ≤1,750 °/s² |

Answer:
- The split is physically sound. The gyro sees only rotation, apart from its g-sensitivity. The accelerometer's dynamic content is mostly linear (86%).
- **Rotation:** reproducing it at real time needs a fast 3-axis gimbal, about 600 °/s and 10,000 °/s², with the IMU at its centre. The bare arm is about 2× too slow in angular acceleration.
- **Linear acceleration:** a 600 mm stick with orientation left free reaches real time on half the trajectories and ≤1.56× on the rest of those tested. The IMU turns with the stick (36–61°, up to about 210 °/s). This is fine for an accelerometer test because forward kinematics gives the ground truth, and the gravity re-projection (up to about 0.8 g) adds coverage.
- **The free orientation has three costs:**
  - The gyro's acceleration sensitivity has to be separated by regression on the known rotation and acceleration, or measured in a few dedicated runs at fixed orientation (slower) or on a constant-rate rig.
  - For composite streams, errors measured in the linear test are re-applied to a sensor orientation up to about 60° away. That is second order for errors that scale with the signal.
  - A counter-rotating tip gimbal (±60°, 210 °/s, 1,750 °/s²) would remove both, but it is optional.
- The first-round verdict that a position-only stick gives the "wrong" rotation and gravity projection judged it against full human fidelity. That does not apply to the split tests.
- **The same gimbal on the stick can follow the human orientation** for full combined runs at the linear test's time scale (gimbal relative rate up to about 600 °/s, range about ±100°). This is untested and is the best route to the cross-effect tests.

Limits: TCP-level IK only, with no walls, no self-collision, no tip-mass dynamics and no stick bending. The joint-acceleration limit was assumed, not measured.

## Caveats & issues
- Capability is a TCP-level proxy: Jacobian box/LP, ignoring J̇q̇ and torque. The washed or warped paths were not IK-checked; next, re-run OV with direction-dependent caps (about 0.6–0.75 m/s, about 0.5 g) and verify with the repo's fitter.
- The orientation-filter results come from one simple filter (Mahony, ideal sensors). Commercial filters with adaptive gating are less sensitive to dynamic acceleration, so the ratios are indicative.
- Two first-round claims were withdrawn after the cross-check:
  - "gyro limit dominant, 160 °/s conservative" holds only if a_cap ≥ 1.7 g;
  - "76% of >1 g bursts fit without slowdown" checked the velocity limit only.
- The 0.4 m washout headline from the first round (O, about 0.25 g error) is for an infeasible trajectory.
- The earlier log attributes the ~2.1× floor to the 1 m/s TCP limit. The joint-speed limits are actually tighter: 0.56–0.97 m/s when the orientation is held.

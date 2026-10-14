# Robot trajectories: export folder

Each subfolder is one **slate**: a batch of trajectories from several participants, with
the identities removed. Each trial appears once per slate, with either its thigh or its
shank trajectory (chosen at random), so the two segments of one trial cannot be paired. Files are named only by a zero-padded index (`0001.csv`, `0002.csv`,
...), in a random order. Nothing in this folder names a participant, session, trial, marker
or segment.

Format: format B (Cartesian) from `docs/Coding_docs/trajectory_format.md`.
- `t` in seconds from 0, sampled at 60 Hz.
- `x,y,z` in mm. The frame is the **capture (lab) frame, not the robot base frame** until
  the lab-to-base transform is supplied. Do not play these files on the robot as they are.
- `roll,pitch,yaw` (rad) are present in every file. They come from a technical marker
  cluster frame, not an anatomical one. Pitch often sits near ±90°, where the angles change
  quickly. Several files exceed the robot's 8° per-row clamp. Treat the orientation as
  unvalidated until it is fixed on the robot side.
- Peak TCP speeds are about 1.5–2 m/s, far above the robot's speed limits. The trajectories
  need time scaling before playback.

The mapping from index to source is kept outside this folder, in a private location that is
excluded from git. Do not copy it over. Each slate has its own random ID.

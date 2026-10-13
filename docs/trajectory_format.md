# Trajectory CSV Format

Playback scripts expect a CSV sampled at a fixed or variable timestep.

## Required columns

- `t`: time in seconds from trajectory start.
- `q1`..`q7`: joint positions in radians.

## Optional columns

- `dq1`..`dq7`: joint velocities in radians/second.

Velocity control mode requires all `dq` columns.

## Example

```csv
t,q1,q2,q3,q4,q5,q6,q7,dq1,dq2,dq3,dq4,dq5,dq6,dq7
0.00,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1
0.02,0.002,0.001,0.0,-0.001,0.001,0.0,0.002,0.1,0.08,0.07,0.05,0.03,0.02,0.01
```

## Tips for recorded trajectories

- Keep timestamps monotonic and starting at `0.0`.
- Use consistent units (radians and radians/second).
- Smooth noisy derivatives before velocity playback.

## Cartesian trajectory format

Cartesian playback uses firmware IK and expects:

- Required: `t,x,y,z`
- Orientation (choose one):
	- `roll,pitch,yaw`
	- `gripper_roll,gripper_pitch,gripper_yaw`
- If orientation columns are omitted, playback uses:
	- `--default-orientation-rpy roll,pitch,yaw` if provided
	- otherwise current robot orientation at run start
- Optional: `vx,vy,vz,vroll,vpitch,vyaw`

### Units

- `x,y,z` in mm
- `roll,pitch,yaw` in radians
- `vx,vy,vz` in mm/s
- `vroll,vpitch,vyaw` in rad/s

### Mode mapping

- `servo` mode: uses `set_servo_cartesian` with pose columns
- `pose` mode: uses `set_position` with pose columns
- `velocity` mode: uses `vc_set_cartesian_velocity` and requires all velocity columns

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class JointTrajectory:
    t: list[float]
    q: list[list[float]]
    dq: list[list[float]] | None


@dataclass
class CartesianTrajectory:
    t: list[float]
    pose: list[list[float]]
    twist: list[list[float]] | None
    orientation_source: str


def load_joint_trajectory_csv(path: str | Path) -> JointTrajectory:
    df = pd.read_csv(path)
    required = ["t"] + [f"q{i}" for i in range(1, 8)]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    t = df["t"].astype(float).tolist()
    q = df[[f"q{i}" for i in range(1, 8)]].astype(float).values.tolist()

    velocity_cols = [f"dq{i}" for i in range(1, 8)]
    has_velocity = all(c in df.columns for c in velocity_cols)
    dq = df[velocity_cols].astype(float).values.tolist() if has_velocity else None

    return JointTrajectory(t=t, q=q, dq=dq)


def load_cartesian_trajectory_csv(path: str | Path) -> CartesianTrajectory:
    df = pd.read_csv(path)
    required = ["t", "x", "y", "z"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    t = df["t"].astype(float).tolist()

    orientation_source = "none"
    if all(c in df.columns for c in ["roll", "pitch", "yaw"]):
        rpy_df = df[["roll", "pitch", "yaw"]].astype(float)
        orientation_source = "rpy"
    elif all(c in df.columns for c in ["gripper_roll", "gripper_pitch", "gripper_yaw"]):
        rpy_df = df[["gripper_roll", "gripper_pitch", "gripper_yaw"]].astype(float)
        rpy_df.columns = ["roll", "pitch", "yaw"]
        orientation_source = "gripper_rpy"
    else:
        # Orientation can be injected later from CLI default or current robot pose.
        rpy_df = pd.DataFrame(
            {
                "roll": [float("nan")] * len(df),
                "pitch": [float("nan")] * len(df),
                "yaw": [float("nan")] * len(df),
            }
        )

    pose_df = pd.concat([df[["x", "y", "z"]].astype(float), rpy_df], axis=1)
    pose = pose_df[["x", "y", "z", "roll", "pitch", "yaw"]].values.tolist()

    twist_cols = ["vx", "vy", "vz", "vroll", "vpitch", "vyaw"]
    has_twist = all(c in df.columns for c in twist_cols)
    twist = df[twist_cols].astype(float).values.tolist() if has_twist else None

    return CartesianTrajectory(
        t=t,
        pose=pose,
        twist=twist,
        orientation_source=orientation_source,
    )

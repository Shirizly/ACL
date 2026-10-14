from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Optional column that groups several trajectories into one file (see docs/trajectory_format.md).
TRAJ_COLUMN = "traj"


def split_trajectory_ref(ref: str | Path) -> tuple[Path, str | None]:
    """'bank.csv#pick_left' -> (Path('bank.csv'), 'pick_left'); a plain path -> (path, None).

    A path that exists as-is is never split, so file names containing '#' still work.
    """
    text = str(ref)
    if Path(text).exists() or "#" not in text:
        return Path(text), None
    path, _, selector = text.rpartition("#")
    return Path(path), selector


def list_trajectories(df: pd.DataFrame) -> list[tuple[str, int]]:
    """(name, row count) per trajectory, in order of first appearance."""
    if TRAJ_COLUMN not in df.columns:
        return []
    counts = df[TRAJ_COLUMN].value_counts(sort=False)
    return [(str(name), int(counts[name])) for name in pd.unique(df[TRAJ_COLUMN])]


def select_trajectory(df: pd.DataFrame, selector: str | None, source: str = "") -> pd.DataFrame:
    """Rows of one trajectory from a (possibly) multi-trajectory frame.

    selector: a name from the 'traj' column (exact match wins), else a 0-based index in
    order of first appearance. A file without the 'traj' column is a single trajectory.
    Columns that are blank for every row of the selected trajectory are dropped.
    """
    if TRAJ_COLUMN not in df.columns:
        if selector is not None:
            raise ValueError(f"{source}: selector '#{selector}' given but the file has no '{TRAJ_COLUMN}' column")
        return df
    names = [n for n, _ in list_trajectories(df)]
    listing = ", ".join(f"[{i}] {n}" for i, n in enumerate(names))
    if df[TRAJ_COLUMN].isna().any():
        raise ValueError(f"{source}: empty '{TRAJ_COLUMN}' cell at row {int(df[TRAJ_COLUMN].isna().to_numpy().argmax())}")
    if selector is None:
        if len(names) == 1:
            selector = names[0]
        else:
            raise ValueError(f"{source} holds {len(names)} trajectories; pick one with '<file>#<name or index>': {listing}")
    if selector in names:
        name = selector
    elif selector.lstrip("-").isdigit() and 0 <= int(selector) < len(names):
        name = names[int(selector)]
    else:
        raise ValueError(f"{source}: no trajectory '{selector}'. Available: {listing}")
    part = df[df[TRAJ_COLUMN] == name].drop(columns=[TRAJ_COLUMN]).reset_index(drop=True)
    # Columns used only by other trajectories in the file are blank here: treat them as absent.
    return part.dropna(axis=1, how="all")


def read_trajectory_frame(ref: str | Path) -> pd.DataFrame:
    """Read 'file.csv' or 'file.csv#selector' and return the selected trajectory's rows."""
    path, selector = split_trajectory_ref(ref)
    df = pd.read_csv(path, dtype={TRAJ_COLUMN: str})
    return select_trajectory(df, selector, source=str(path))


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
    df = read_trajectory_frame(path)
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
    df = read_trajectory_frame(path)
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


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="List the trajectories in a trajectory CSV")
    p.add_argument("csv")
    args = p.parse_args()
    path, _ = split_trajectory_ref(args.csv)
    df = pd.read_csv(path, dtype={TRAJ_COLUMN: str})
    entries = list_trajectories(df)
    if not entries:
        print(f"{path}: single trajectory, {len(df)} rows (no '{TRAJ_COLUMN}' column)")
    for i, (name, rows) in enumerate(entries):
        part = df[df[TRAJ_COLUMN] == name]
        print(f"[{i}] {name}: {rows} rows, t = {part['t'].iloc[0]:.3f}..{part['t'].iloc[-1]:.3f} s   -> {path}#{name}")


if __name__ == "__main__":
    main()

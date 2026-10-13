from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_sample(duration_s: float, dt: float) -> pd.DataFrame:
    t = np.arange(0.0, duration_s + 1e-9, dt)

    x0, y0, z0 = 300.0, 0.0, 250.0
    amp_mm = 30.0
    freq_hz = 0.1

    x = x0 + amp_mm * np.sin(2.0 * np.pi * freq_hz * t)
    y = y0 + amp_mm * np.cos(2.0 * np.pi * freq_hz * t)
    z = z0 + 10.0 * np.sin(2.0 * np.pi * freq_hz * t)

    roll = np.zeros_like(t)
    pitch = np.pi * np.ones_like(t)
    yaw = np.zeros_like(t)

    vx = amp_mm * 2.0 * np.pi * freq_hz * np.cos(2.0 * np.pi * freq_hz * t)
    vy = -amp_mm * 2.0 * np.pi * freq_hz * np.sin(2.0 * np.pi * freq_hz * t)
    vz = 10.0 * 2.0 * np.pi * freq_hz * np.cos(2.0 * np.pi * freq_hz * t)
    vroll = np.zeros_like(t)
    vpitch = np.zeros_like(t)
    vyaw = np.zeros_like(t)

    return pd.DataFrame(
        {
            "t": t,
            "x": x,
            "y": y,
            "z": z,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "vroll": vroll,
            "vpitch": vpitch,
            "vyaw": vyaw,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sample xArm Cartesian trajectory CSV"
    )
    parser.add_argument("--out", default="data/trajectories/sample_cartesian_circle.csv")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.02)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = build_sample(args.duration, args.dt)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} samples to {out}")


if __name__ == "__main__":
    main()

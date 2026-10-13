from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_sample(duration_s: float, dt: float) -> pd.DataFrame:
    t = np.arange(0.0, duration_s + 1e-9, dt)

    freq_hz = 0.2
    amp = 0.25
    phase = np.linspace(0.0, 1.2, 7)

    q = np.array([
        amp * np.sin(2.0 * np.pi * freq_hz * t + ph) for ph in phase
    ]).T
    dq = np.array([
        amp * 2.0 * np.pi * freq_hz * np.cos(2.0 * np.pi * freq_hz * t + ph) for ph in phase
    ]).T

    data = {"t": t}
    for i in range(7):
        data[f"q{i+1}"] = q[:, i]
    for i in range(7):
        data[f"dq{i+1}"] = dq[:, i]

    return pd.DataFrame(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample xArm 7 joint trajectory CSV")
    parser.add_argument("--out", default="data/trajectories/sample_sine_joint.csv")
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

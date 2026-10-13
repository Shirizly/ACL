#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT="${1:-data/trajectories/cartesian_bigmove.csv}"
export OUT

python - <<'PY'
import os
import numpy as np
import pandas as pd
from pathlib import Path

out = Path(os.environ["OUT"])
out.parent.mkdir(parents=True, exist_ok=True)

# Large visible motion in simulator view.
dt = 0.05
t = np.arange(0.0, 20.0 + 1e-9, dt)

x = np.piecewise(
    t,
    [t < 6.5, (t >= 6.5) & (t < 13.0), t >= 13.0],
    [
        lambda s: 220 + (420 - 220) * (s / 6.5),
        lambda s: 420 + (260 - 420) * ((s - 6.5) / 6.5),
        lambda s: 260 + (340 - 260) * ((s - 13.0) / 7.0),
    ],
)
y = np.piecewise(
    t,
    [t < 10.0, t >= 10.0],
    [lambda s: -180 + 220 * (s / 10.0), lambda s: 40 - 200 * ((s - 10.0) / 10.0)],
)
z = 180 + 80 * np.sin(2 * np.pi * 0.08 * t)

roll = np.pi * np.ones_like(t)
pitch = np.zeros_like(t)
yaw = 0.5 * np.sin(2 * np.pi * 0.05 * t)

df = pd.DataFrame({
    "t": t,
    "x": x,
    "y": y,
    "z": z,
    "roll": roll,
    "pitch": pitch,
    "yaw": yaw,
})

df.to_csv(out, index=False)
print(f"Wrote {len(df)} samples to {out}")
PY

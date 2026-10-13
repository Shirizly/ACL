from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _interpolate_commanded_xyz(
    commanded: pd.DataFrame,
    t_meas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "record_t" in commanded.columns:
        t_cmd = commanded["record_t"].astype(float).to_numpy()
    elif "t_sent" in commanded.columns:
        t_cmd = commanded["t_sent"].astype(float).to_numpy()
    elif "t" in commanded.columns:
        t_cmd = commanded["t"].astype(float).to_numpy()
    else:
        raise ValueError("Commanded file must include t_sent or t column")

    for col in ["x", "y", "z"]:
        if col not in commanded.columns:
            raise ValueError(f"Commanded file missing column: {col}")

    x_cmd = commanded["x"].astype(float).to_numpy()
    y_cmd = commanded["y"].astype(float).to_numpy()
    z_cmd = commanded["z"].astype(float).to_numpy()

    x_ref = np.interp(t_meas, t_cmd, x_cmd, left=x_cmd[0], right=x_cmd[-1])
    y_ref = np.interp(t_meas, t_cmd, y_cmd, left=y_cmd[0], right=y_cmd[-1])
    z_ref = np.interp(t_meas, t_cmd, z_cmd, left=z_cmd[0], right=z_cmd[-1])
    return x_ref, y_ref, z_ref


def analyze_cartesian_error(
    commanded_path: str | Path,
    measured_path: str | Path,
    out_dir: str | Path,
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    commanded = pd.read_csv(commanded_path)
    measured = pd.read_csv(measured_path)

    if measured.empty:
        raise ValueError("Measured data is empty, cannot analyze error")

    for col in ["t", "x", "y", "z"]:
        if col not in measured.columns:
            raise ValueError(f"Measured file missing column: {col}")

    t_meas = measured["t"].astype(float).to_numpy()
    x_meas = measured["x"].astype(float).to_numpy()
    y_meas = measured["y"].astype(float).to_numpy()
    z_meas = measured["z"].astype(float).to_numpy()

    x_ref, y_ref, z_ref = _interpolate_commanded_xyz(commanded, t_meas)

    ex = x_meas - x_ref
    ey = y_meas - y_ref
    ez = z_meas - z_ref

    abs_ex = np.abs(ex)
    abs_ey = np.abs(ey)
    abs_ez = np.abs(ez)
    abs_norm = np.sqrt(ex * ex + ey * ey + ez * ez)

    error_df = pd.DataFrame(
        {
            "t": t_meas,
            "x_cmd": x_ref,
            "y_cmd": y_ref,
            "z_cmd": z_ref,
            "x_meas": x_meas,
            "y_meas": y_meas,
            "z_meas": z_meas,
            "ex": ex,
            "ey": ey,
            "ez": ez,
            "abs_ex": abs_ex,
            "abs_ey": abs_ey,
            "abs_ez": abs_ez,
            "abs_e_norm": abs_norm,
        }
    )

    metrics = {
        "mean_abs_e_norm_mm": float(np.mean(abs_norm)),
        "max_abs_e_norm_mm": float(np.max(abs_norm)),
        "rmse_x_mm": float(np.sqrt(np.mean(ex * ex))),
        "rmse_y_mm": float(np.sqrt(np.mean(ey * ey))),
        "rmse_z_mm": float(np.sqrt(np.mean(ez * ez))),
        "mean_abs_ex_mm": float(np.mean(abs_ex)),
        "mean_abs_ey_mm": float(np.mean(abs_ey)),
        "mean_abs_ez_mm": float(np.mean(abs_ez)),
        "samples": int(len(error_df)),
    }

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(t_meas, abs_norm, linewidth=1.0, color="#1f77b4")
    axes[0].set_ylabel("Absolute Error Norm [mm]")
    axes[0].set_title("Cartesian Tracking Error")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_meas, ex, linewidth=0.9, label="ex")
    axes[1].plot(t_meas, ey, linewidth=0.9, label="ey")
    axes[1].plot(t_meas, ez, linewidth=0.9, label="ez")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Coordinate Error [mm]")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    figure_path = out / "cartesian_tracking_error.png"
    error_csv_path = out / "cartesian_error_timeseries.csv"
    metrics_json_path = out / "cartesian_error_metrics.json"

    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    error_df.to_csv(error_csv_path, index=False)
    metrics_json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "figure_path": str(figure_path),
        "error_csv_path": str(error_csv_path),
        "metrics_json_path": str(metrics_json_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze commanded vs measured Cartesian trajectory error"
    )
    parser.add_argument("--commanded", required=True, help="Path to commanded CSV")
    parser.add_argument("--measured", required=True, help="Path to measured CSV")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_cartesian_error(
        commanded_path=args.commanded,
        measured_path=args.measured,
        out_dir=args.out_dir,
    )
    print(f"saved_error_figure: {result['figure_path']}")
    print(f"saved_error_csv: {result['error_csv_path']}")
    print(f"saved_error_metrics: {result['metrics_json_path']}")


if __name__ == "__main__":
    main()

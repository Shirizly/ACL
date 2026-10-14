"""Load and validate config/safety.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "safety.yaml"

LINK_NAMES = (
    "base",
    "upper_arm",
    "elbow",
    "elbow_offset",
    "forearm",
    "wrist",
    "wrist_offset",
    "tool",
)


@dataclass
class PreprocessConfig:
    enabled: bool = True
    policy: str = "reject"
    interp_resolution_mm: float = 5.0
    interp_resolution_deg: float = 1.0
    ik_model_tolerance_mm: float = 10.0
    clip_smoothing_s: float = 0.3


@dataclass
class RuntimeConfig:
    enabled: bool = True
    command_check: bool = True
    monitor_rate_hz: float = 250.0
    stop_margin_mm: float = 5.0
    latency_s: float = 0.02
    stopping_time_s: float = 0.08
    velocity_window_s: float = 0.02

    @property
    def lookahead_s(self) -> float:
        return self.latency_s + self.stopping_time_s


@dataclass
class FenceConfig:
    enabled: bool = True
    shrink_mm: float | None = None  # None = auto
    restore_on_exit: bool = True


@dataclass
class SafetyConfig:
    depth_D_mm: float
    width_W_mm: float
    height_H_mm: float
    base_x_mm: float = 0.0
    base_y_mm: float = 0.0
    base_z_mm: float = 0.0
    base_yaw_deg: float = 0.0
    safety_margin_mm: float = 20.0
    tool_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radii_mm: dict[str, float] = field(default_factory=dict)
    firmware_tcp_offset: tuple[float, ...] = (0.0,) * 6  # mm, deg
    # Optional rigid box on the gripper (e.g. an attachment): ((x0,x1),(y0,y1),(z0,z1)) mm in tool_box_frame.
    tool_box: tuple[tuple[float, float], ...] | None = None
    tool_box_frame: str = "flange"  # 'flange' (J7 face) or 'tool' (flange frame moved to tool_offset_mm)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    firmware_fence: FenceConfig = field(default_factory=FenceConfig)
    source_path: str = ""

    def tool_box_corners_mm(self):
        """(8,3) corners of the gripper box in the FLANGE frame, corner i = 4*ix + 2*iy + iz; None if unset."""
        if self.tool_box is None:
            return None
        import itertools

        import numpy as np

        (x0, x1), (y0, y1), (z0, z1) = self.tool_box
        c = np.array(list(itertools.product((x0, x1), (y0, y1), (z0, z1))), dtype=float)
        if self.tool_box_frame == "tool":
            c = c + np.asarray(self.tool_offset_mm, dtype=float)
        return c

    @property
    def firmware_tcp_offset_rad(self) -> list[float]:
        o = list(self.firmware_tcp_offset)
        return o[:3] + [math.radians(v) for v in o[3:6]]

    @property
    def fence_shrink_mm(self) -> float:
        if self.firmware_fence.shrink_mm is not None:
            return float(self.firmware_fence.shrink_mm)
        # The fence watches the firmware TCP point; in any valid motion the flange keeps at least the
        # wrist_offset capsule radius (+ margin) from the walls, and the tool capsule its own radius.
        return self.safety_margin_mm + max(float(self.radii_mm["tool"]), float(self.radii_mm["wrist_offset"]))


def _req(d: dict, key: str, ctx: str):
    if key not in d:
        raise ValueError(f"safety config: missing '{ctx}.{key}'")
    return d[key]


WALL_KEYS = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


def walls_in_base_to_box(walls: dict, has_box_fields: bool = False) -> tuple[dict, dict]:
    """Alternative workcell format: each wall's coordinate along a ROBOT BASE axis (the frame of
    get_position(); z = 0 at the base mounting surface). Walls are parallel to the base axes.

        workcell:
          walls_in_base_mm: {x_min: -500, x_max: 650, y_min: -400, y_max: 400, z_min: 20, z_max: 1180}

    Converted to the box form: D,W,H plus the base pose in the box-floor frame (yaw 0)."""
    if has_box_fields:
        raise ValueError("safety config: give either workcell.walls_in_base_mm or the box form "
                         "(depth_D_mm/width_W_mm/height_H_mm + robot_base_in_world), not both")
    missing = [k for k in WALL_KEYS if k not in walls]
    if missing:
        raise ValueError(f"safety config: workcell.walls_in_base_mm missing {missing}")
    w = {k: float(walls[k]) for k in WALL_KEYS}
    for a in "xyz":
        if w[f"{a}_max"] <= w[f"{a}_min"]:
            raise ValueError(f"safety config: walls_in_base_mm.{a}_max must be > {a}_min")
    box = {"depth_D_mm": w["x_max"] - w["x_min"], "width_W_mm": w["y_max"] - w["y_min"],
           "height_H_mm": w["z_max"] - w["z_min"]}
    # box frame: origin at the center of the box floor -> the base sits at minus that point
    base = {"x_mm": -(w["x_min"] + w["x_max"]) / 2, "y_mm": -(w["y_min"] + w["y_max"]) / 2,
            "z_mm": -w["z_min"], "yaw_deg": 0.0}
    return box, base


def load_safety_config(path: str | Path | None = None) -> SafetyConfig:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    wc = _req(raw, "workcell", "")
    base = raw.get("robot_base_in_world", {}) or {}
    if "walls_in_base_mm" in wc:
        wc, base = walls_in_base_to_box(wc["walls_in_base_mm"], has_box_fields=any(
            k in wc for k in ("depth_D_mm", "width_W_mm", "height_H_mm")) or bool(raw.get("robot_base_in_world")))
    geom = raw.get("geometry", {}) or {}
    radii = {k: float(v) for k, v in (geom.get("radii_mm") or {}).items()}
    missing = [n for n in LINK_NAMES if n not in radii]
    if missing:
        raise ValueError(f"safety config: geometry.radii_mm missing {missing}")
    if geom.get("dh", "nominal") != "nominal":
        raise ValueError("safety config: only geometry.dh: nominal is supported")

    tb = geom.get("tool_box")
    tool_box = None
    tool_box_frame = "flange"
    if tb:
        tool_box = tuple((float(tb[k][0]), float(tb[k][1])) for k in ("x_mm", "y_mm", "z_mm"))
        tool_box_frame = str(tb.get("frame", "flange"))
    # The box is checked like a link with this radius (0 = the exact box).
    radii["tool_box"] = float((tb or {}).get("inflate_mm", 0.0))

    pp = raw.get("preprocess", {}) or {}
    rt = raw.get("runtime", {}) or {}
    fw = raw.get("firmware_fence", {}) or {}
    shrink = fw.get("shrink_mm", "auto")

    cfg = SafetyConfig(
        depth_D_mm=float(_req(wc, "depth_D_mm", "workcell")),
        width_W_mm=float(_req(wc, "width_W_mm", "workcell")),
        height_H_mm=float(_req(wc, "height_H_mm", "workcell")),
        base_x_mm=float(base.get("x_mm", 0.0)),
        base_y_mm=float(base.get("y_mm", 0.0)),
        base_z_mm=float(base.get("z_mm", 0.0)),
        base_yaw_deg=float(base.get("yaw_deg", 0.0)),
        safety_margin_mm=float(raw.get("safety_margin_mm", 20.0)),
        tool_offset_mm=tuple(float(v) for v in geom.get("tool_offset_mm", [0, 0, 0])),  # type: ignore[arg-type]
        radii_mm=radii,
        firmware_tcp_offset=tuple(float(v) for v in geom.get("firmware_tcp_offset", [0] * 6)),
        tool_box=tool_box,
        tool_box_frame=tool_box_frame,
        preprocess=PreprocessConfig(**pp),
        runtime=RuntimeConfig(**rt),
        firmware_fence=FenceConfig(
            enabled=bool(fw.get("enabled", True)),
            shrink_mm=None if shrink in (None, "auto") else float(shrink),
            restore_on_exit=bool(fw.get("restore_on_exit", True)),
        ),
        source_path=str(path),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: SafetyConfig) -> None:
    for name in ("depth_D_mm", "width_W_mm", "height_H_mm"):
        if getattr(cfg, name) <= 0:
            raise ValueError(f"safety config: {name} must be > 0")
    if cfg.safety_margin_mm < 0:
        raise ValueError("safety config: safety_margin_mm must be >= 0")
    if cfg.preprocess.policy not in ("reject", "clip"):
        raise ValueError("safety config: preprocess.policy must be 'reject' or 'clip'")
    if cfg.preprocess.interp_resolution_mm <= 0 or cfg.preprocess.interp_resolution_deg <= 0:
        raise ValueError("safety config: interp resolutions must be > 0")
    if cfg.runtime.monitor_rate_hz <= 0:
        raise ValueError("safety config: runtime.monitor_rate_hz must be > 0")
    if cfg.tool_box is not None:
        if any(lo > hi for lo, hi in cfg.tool_box):
            raise ValueError("safety config: geometry.tool_box ranges must be [min, max]")
        if cfg.tool_box_frame not in ("flange", "tool"):
            raise ValueError("safety config: geometry.tool_box.frame must be 'flange' or 'tool'")
    if len(cfg.tool_offset_mm) != 3 or len(cfg.firmware_tcp_offset) != 6:
        raise ValueError("safety config: tool_offset_mm needs 3 values, firmware_tcp_offset 6")

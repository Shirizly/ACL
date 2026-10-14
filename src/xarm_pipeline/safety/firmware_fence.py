"""Layer 3: xArm firmware safety boundary ("fence").

Measured behaviour (xarm-python-sdk 1.18.4, simulator firmware v2.4.0):

* ``set_reduced_tcp_boundary([x_max, x_min, y_max, y_min, z_max, z_min])``
  (mm, integers, ROBOT BASE frame, axis-aligned) + ``set_fence_mode(True)``.
  ``get_reduced_states()[1]`` reads the boundary back; element [5] is the
  fence on/off flag.
* Only the firmware TCP *point* is checked; an elbow 137 mm above a z_max
  boundary did not trigger it.
* It STOPS, it does not clamp: the arm halts at the boundary (state 4).
  set_position / set_servo_angle(wait=True) return -9 with no error code;
  servo (mode 1) and velocity (modes 4, 5) streaming raise ControllerError
  C35 ("safety boundary limit").  Recovery: clean_error(), motion_enable(True),
  set_mode(...), set_state(0); moving back inside is allowed.
* Not persisted unless ``save_conf()`` is called (this module never calls it);
  after a simulator restart the boundary was back to +-9999 and fence off.

CLI::

    python -m src.xarm_pipeline.safety.firmware_fence --ip 127.0.0.1 status
    python -m src.xarm_pipeline.safety.firmware_fence --ip 127.0.0.1 apply
    python -m src.xarm_pipeline.safety.firmware_fence --ip 127.0.0.1 reset
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from .config import SafetyConfig, load_safety_config

DISABLED_BOUNDARY = [9999, -9999, 9999, -9999, 9999, -9999]


def inscribed_base_box(cfg: SafetyConfig) -> list[int]:
    """Base-frame axis-aligned box inscribed in the world box shrunk by
    ``cfg.fence_shrink_mm``. Returns [x_max, x_min, y_max, y_min, z_max, z_min]
    as integers (rounded inward)."""
    s = cfg.fence_shrink_mm
    a = cfg.depth_D_mm / 2 - s  # half extents of the shrunk world box
    b = cfg.width_W_mm / 2 - s
    if a <= 0 or b <= 0 or cfg.height_H_mm - 2 * s <= 0:
        raise ValueError("firmware fence: box too small for the shrink value")
    yaw = math.radians(cfg.base_yaw_deg)
    # World box center expressed in the base frame.
    c, sn = math.cos(yaw), math.sin(yaw)
    dx, dy = -cfg.base_x_mm, -cfg.base_y_mm
    cx = c * dx + sn * dy
    cy = -sn * dx + c * dy
    # Axis-aligned (in base) rectangle, half extents (u,v), centered at (cx,cy),
    # inside a rectangle rotated by -yaw: u|c|+v|s| <= a and u|s|+v|c| <= b.
    ac, as_ = abs(c), abs(sn)
    best = (0.0, 0.0)
    if as_ < 1e-9 or ac < 1e-9:
        best = (a, b) if as_ < 1e-9 else (b, a)
    else:
        for u in np.linspace(0.0, min(a / ac, b / as_), 2001):
            v = min((a - u * ac) / as_, (b - u * as_) / ac)
            if v > 0 and u * v > best[0] * best[1]:
                best = (u, v)
    u, v = best
    z_lo = s - cfg.base_z_mm
    z_hi = cfg.height_H_mm - s - cfg.base_z_mm
    return [
        int(math.floor(cx + u)), int(math.ceil(cx - u)),
        int(math.floor(cy + v)), int(math.ceil(cy - v)),
        int(math.floor(z_hi)), int(math.ceil(z_lo)),
    ]


def read_state(arm) -> dict:
    code, st = arm.get_reduced_states()
    if code != 0:
        raise RuntimeError(f"get_reduced_states failed: code={code}")
    return {"reduced_mode": st[0], "boundary": list(st[1]), "fence_on": int(st[5]) if len(st) > 5 else None}


def apply_fence(arm, cfg: SafetyConfig) -> dict:
    """Set the boundary and enable the fence. Returns the previous state."""
    prev = read_state(arm)
    box = inscribed_base_box(cfg)
    code = arm.set_reduced_tcp_boundary(box)
    if code != 0:
        raise RuntimeError(f"set_reduced_tcp_boundary failed: code={code}")
    ret = arm.set_fence_mode(True)
    code = ret[0] if isinstance(ret, (list, tuple)) else ret
    if code != 0:
        raise RuntimeError(f"set_fence_mode failed: code={code}")
    print(f"firmware fence ON: base-frame TCP box [x_max,x_min,y_max,y_min,z_max,z_min] = {box} mm "
          f"(previous: boundary={prev['boundary']} fence_on={prev['fence_on']})", flush=True)
    return prev


def restore_fence(arm, prev: dict) -> None:
    arm.set_reduced_tcp_boundary(prev["boundary"])
    arm.set_fence_mode(bool(prev["fence_on"]))
    print(f"firmware fence restored: boundary={prev['boundary']} fence_on={prev['fence_on']}", flush=True)


def reset_fence(arm) -> None:
    arm.set_fence_mode(False)
    arm.set_reduced_tcp_boundary(DISABLED_BOUNDARY)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Inspect / set / reset the xArm firmware safety boundary")
    p.add_argument("--ip", default="127.0.0.1")
    p.add_argument("--config", default=None)
    p.add_argument("action", choices=["status", "apply", "reset", "compute"])
    args = p.parse_args(argv)
    cfg = load_safety_config(args.config)
    if args.action == "compute":
        print(f"fence box (base frame) = {inscribed_base_box(cfg)}")
        return
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(args.ip, baud_checkset=False, check_joint_limit=False)
    try:
        if args.action == "apply":
            apply_fence(arm, cfg)
        elif args.action == "reset":
            reset_fence(arm)
        print(f"firmware fence state: {read_state(arm)}")
        print(f"config-derived fence box would be: {inscribed_base_box(cfg)}")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()

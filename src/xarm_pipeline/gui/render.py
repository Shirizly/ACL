"""3D render of the xArm 7 (capsule model) and the safety box, embeddable in Qt (pyvistaqt).

Mouse: left-drag rotates, right-drag/wheel zooms, middle-drag pans (VTK native).
"""
from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from src.xarm_pipeline.safety.config import SafetyConfig
from src.xarm_pipeline.safety.geometry import CAPSULES, Workcell, chain_points_batch

READY_Q_DEG = (0.0, -28.17, 0.0, 23.24, 0.0, 51.41, 0.0)
# Box edges for the 8 corners produced by tool_box_corners_mm() (bit order x,y,z).
_BOX_FACES = np.array([[4, 0, 1, 3, 2], [4, 4, 5, 7, 6], [4, 0, 1, 5, 4],
                       [4, 2, 3, 7, 6], [4, 0, 2, 6, 4], [4, 1, 3, 7, 5]]).ravel()


class RobotScene:
    def __init__(self, parent, cfg: SafetyConfig):
        self.plotter = QtInteractor(parent)
        self.plotter.set_background("white")
        self.plotter.add_axes()
        self._actors_arm: list = []
        self.set_config(cfg)
        self.set_joints(np.radians(READY_Q_DEG))
        self.plotter.view_isometric()

    @property
    def widget(self):
        return self.plotter

    def set_config(self, cfg: SafetyConfig) -> None:
        self.wc = Workcell(cfg)
        p = self.plotter
        for name in ("walls", "wall_edges", "floor"):
            p.remove_actor(name)
        lo, hi = self.wc.lo, self.wc.hi
        box = pv.Box(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
        self.wall_actors = [
            p.add_mesh(box, color="tomato", opacity=0.12, name="walls"),
            p.add_mesh(box.extract_feature_edges(), color="red", line_width=2, name="wall_edges"),
        ]
        floor = pv.Plane(center=(0, 0, 0), i_size=hi[0] - lo[0], j_size=hi[1] - lo[1])
        p.add_mesh(floor, color="lightgray", opacity=0.5, name="floor")
        self._build_arm()

    def _build_arm(self) -> None:
        for a in self._actors_arm:
            self.plotter.remove_actor(a)
        self._actors_arm = []
        self._segs: list[tuple[pv.PolyData, int, int, float]] = []
        for name, i, j in CAPSULES:
            if name == "tool_box":
                continue
            r = max(float(self.wc.cfg.radii_mm.get(name, 0.0)), 8.0)
            mesh = pv.Tube(pointa=(0, 0, 0), pointb=(0, 0, 1), radius=r, n_sides=20)
            self._segs.append((mesh, i, j, r))
            color = "dimgray" if name == "base" else "lightsteelblue"
            self._actors_arm.append(self.plotter.add_mesh(mesh, color=color, smooth_shading=True))
        self._joints = pv.PolyData(np.zeros((9, 3)))
        self._actors_arm.append(self.plotter.add_mesh(self._joints, color="steelblue",
                                                      point_size=18, render_points_as_spheres=True))
        self._tool = None
        if self.wc.tool_box_corners is not None:
            self._tool = pv.PolyData(np.zeros((8, 3)), faces=_BOX_FACES)
            self._actors_arm.append(self.plotter.add_mesh(self._tool, color="orange", opacity=0.8))

    def set_joints(self, q_rad) -> None:
        P = chain_points_batch(np.asarray(q_rad, float)[None, :], self.wc.tool_offset,
                               tool_box_corners=self.wc.tool_box_corners)[0]
        Pw = self.wc.base_to_world(P)
        for mesh, i, j, r in self._segs:
            a, b = Pw[i], Pw[j]
            if np.linalg.norm(b - a) < 1e-6:
                b = a + np.array([0, 0, 1e-3])
            mesh.copy_from(pv.Tube(pointa=a, pointb=b, radius=r, n_sides=20))
        self._joints.points = Pw[:9]
        if self._tool is not None:
            self._tool.points = Pw[9:17]
        self.plotter.render()

    def set_walls_visible(self, visible: bool) -> None:
        for a in self.wall_actors:
            a.SetVisibility(visible)
        self.plotter.render()

    def screenshot(self) -> np.ndarray:
        return self.plotter.screenshot(return_img=True)

    def close(self) -> None:
        self.plotter.close()

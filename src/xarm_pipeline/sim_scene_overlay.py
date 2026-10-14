"""Draw the safety workcell (and simple environment boxes) into the UFACTORY Studio 3D view.

Purely visual: it injects three.js objects into the Studio page via Playwright, using
the Studio's own scene (Vue component ``Arm3D`` -> ``vm.three.scene``). Nothing is sent
to the simulator firmware, and the objects take no part in any safety check.

Studio scene convention (measured): 1 unit = 100 mm, three.js Y is up, robot base origin
at (0, -5, 0), and robot base-frame (X, Y, Z) maps to three.js (-X, Z, Y) / 100.
"""

from __future__ import annotations

import itertools
import math
import time
from pathlib import Path

import numpy as np
import yaml

from .safety.config import DEFAULT_CONFIG_PATH, load_safety_config


def _box_corners_world(center, size, yaw_deg) -> np.ndarray:
    """8 corners (world frame, mm) of a box; corner i = 4*ix + 2*iy + iz."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    half = np.asarray(size, float) / 2.0
    local = np.array([[sx * half[0], sy * half[1], sz * half[2]] for sx, sy, sz in itertools.product((-1, 1), repeat=3)])
    return local @ rot.T + np.asarray(center, float)


_FACES = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def _faces_to_triangles(corners: np.ndarray) -> list[float]:
    tris = []
    for a, b, c, d in _FACES:
        tris += [corners[a], corners[b], corners[c], corners[a], corners[c], corners[d]]
    return np.asarray(tris).ravel().round(2).tolist()


def _edge_prisms(corners: np.ndarray, thickness_mm: float) -> list[float]:
    """Each box edge as a thin square prism, so edges show without line primitives."""
    tris: list[float] = []
    for a, b in _EDGES:
        p0, p1 = corners[a], corners[b]
        axis = (p1 - p0) / np.linalg.norm(p1 - p0)
        helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(axis, helper)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        h = thickness_mm / 2.0
        prism = np.array(
            [p + su * h * u + sv * h * v for p in (p0, p1) for su in (-1, 1) for sv in (-1, 1)]
        )  # index = 4*end + 2*su + sv, same bit layout as box corners
        tris += _faces_to_triangles(prism)
    return tris


def build_overlay_spec(config_path: str | Path | None = None, show_margin: bool | None = None) -> dict:
    """Triangle meshes in the robot BASE frame (mm) for the walls, margin and visualization.objects."""
    cfg = load_safety_config(config_path)
    raw = yaml.safe_load(Path(config_path or DEFAULT_CONFIG_PATH).read_text(encoding="utf-8")) or {}
    vis = raw.get("visualization", {}) or {}
    if show_margin is None:
        show_margin = bool(vis.get("show_margin", True))

    yaw = math.radians(cfg.base_yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    rot_wb = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])  # base -> world rotation
    base_pos = np.array([cfg.base_x_mm, cfg.base_y_mm, cfg.base_z_mm])

    def to_base(corners_world: np.ndarray) -> np.ndarray:
        return (corners_world - base_pos) @ rot_wb

    wall_color = str(vis.get("wall_color", "#ff4d4d"))
    edge_mm = float(vis.get("edge_thickness_mm", 8.0))
    D, W, H = cfg.depth_D_mm, cfg.width_W_mm, cfg.height_H_mm
    walls = to_base(_box_corners_world([0, 0, H / 2], [D, W, H], 0.0))
    meshes = [
        {"name": "walls", "tris": _faces_to_triangles(walls), "color": wall_color, "opacity": float(vis.get("wall_opacity", 0.08))},
        {"name": "wall_edges", "tris": _edge_prisms(walls, edge_mm), "color": wall_color, "opacity": 1.0},
    ]
    if show_margin and cfg.safety_margin_mm > 0:
        m = cfg.safety_margin_mm
        margin = to_base(_box_corners_world([0, 0, H / 2], [D - 2 * m, W - 2 * m, H - 2 * m], 0.0))
        meshes.append({"name": "margin_edges", "tris": _edge_prisms(margin, edge_mm / 2), "color": wall_color, "opacity": 0.35})
    for obj in vis.get("objects", []) or []:
        corners = to_base(_box_corners_world(obj["center_mm"], obj["size_mm"], float(obj.get("yaw_deg", 0.0))))
        color = str(obj.get("color", "#8899aa"))
        meshes.append({"name": str(obj.get("name", "object")), "tris": _faces_to_triangles(corners), "color": color, "opacity": float(obj.get("opacity", 0.6))})
        meshes.append({"name": str(obj.get("name", "object")) + "_edges", "tris": _edge_prisms(corners, edge_mm / 2), "color": color, "opacity": 1.0})
    # Gripper geometry, drawn on the arm itself (attached to Studio's last-link group, so it moves with
    # the arm): the 'tool' capsule as a square prism and the attachment box (geometry.tool_box).
    attached = []
    if bool(vis.get("show_tool", True)):
        L = float(np.linalg.norm(cfg.tool_offset_mm))
        r = float(cfg.radii_mm["tool"])
        if L > 0 and r > 0:  # radius 0: the gripper is described by geometry.tool_box instead
            prism = _box_corners_world([0.0, 0.0, L / 2], [2 * r, 2 * r, L], 0.0)
            attached.append({"name": "tool", "tris": _faces_to_triangles(prism),
                             "color": str(vis.get("tool_color", "#c9cdd4")), "opacity": 0.35})
        corners = cfg.tool_box_corners_mm()
        if corners is not None:
            color = str(vis.get("tool_box_color", "#ffb000"))
            # translucent, so a gripper mesh drawn by Studio itself stays visible inside it for comparison
            attached.append({"name": "tool_box", "tris": _faces_to_triangles(corners), "color": color,
                             "opacity": float(vis.get("tool_box_opacity", 0.3))})
            attached.append({"name": "tool_box_edges", "tris": _edge_prisms(corners, edge_mm / 2), "color": color,
                             "opacity": 1.0})
    return {"meshes": meshes, "attached": attached}


# Runs in the page. Builds three.js meshes from constructors borrowed from the robot model
# (Studio does not expose THREE globally and has no line objects in the scene).
_JS_ADD_MESHES = r"""
(spec) => {
  let vm = window.__arm3dVm || null;  // cached: the canvas is later moved out of the component
  for (let el = document.querySelector('canvas'); !vm && el; el = el.parentElement) { if (el.__vue__ && el.__vue__.$options.name === 'Arm3D') vm = el.__vue__; }
  window.__arm3dVm = vm;
  if (!vm || !vm.three || !vm.three.scene) return 'Arm3D scene not found';
  const scene = vm.three.scene;
  let anyMesh = null; scene.traverse(o => { if (!anyMesh && o.isMesh && o.geometry && o.geometry.attributes) anyMesh = o; });
  if (!anyMesh) return 'robot meshes not loaded yet';
  const Mesh = anyMesh.constructor;
  const Geometry = anyMesh.geometry.constructor;
  const Attr = anyMesh.geometry.attributes.position.constructor;
  const Mat = (Array.isArray(anyMesh.material) ? anyMesh.material[0] : anyMesh.material).constructor;
  const setAttr = (g, n, a) => (g.setAttribute ? g.setAttribute(n, a) : g.addAttribute(n, a));

  (vm.__overlay || []).forEach(o => o.parent && o.parent.remove(o));
  vm.__overlay = [];
  for (const m of spec.meshes) {
    const src = m.tris, pos = new Float32Array(src.length);
    // robot base frame (mm) -> Studio scene units: (X, Y, Z) -> (-X, Z, Y)/100, base at y = -5
    for (let i = 0; i < src.length; i += 3) { pos[i] = -src[i] / 100; pos[i + 1] = src[i + 2] / 100 - 5; pos[i + 2] = src[i + 1] / 100; }
    const g = new Geometry(); setAttr(g, 'position', new Attr(pos, 3));
    if (g.computeVertexNormals) g.computeVertexNormals();
    const mat = new Mat({ transparent: m.opacity < 1, opacity: m.opacity, side: 2, depthWrite: m.opacity >= 1 });
    if (mat.color) mat.color.set(m.color);
    if (mat.emissive) mat.emissive.set(m.color);
    const mesh = new Mesh(g, mat); mesh.name = 'overlay_' + m.name; mesh.renderOrder = m.opacity < 1 ? 20 : 0;
    scene.add(mesh); vm.__overlay.push(mesh);
  }
  // Meshes given in the FLANGE frame (mm), attached to the J7 link group of the Studio model. The fixed
  // transform flange(mm) -> group local was measured in the sim (constant across poses to 0.2 mm).
  const groups = vm.curThreeModel && vm.curThreeModel.GROUPS;
  const link7 = groups && groups[7];
  let attached = 0;
  if (link7 && spec.attached && spec.attached.length) {
    const Matrix4 = link7.matrix.constructor;
    const F = new Matrix4();
    F.set(0, 0.01, 0, 0,   0, 0, -0.01, 0,   -0.01, 0, 0, 0,   0, 0, 0, 1);
    for (const m of spec.attached) {
      const g = new Geometry(); setAttr(g, 'position', new Attr(new Float32Array(m.tris), 3));
      if (g.computeVertexNormals) g.computeVertexNormals();
      const mat = new Mat({ transparent: m.opacity < 1, opacity: m.opacity, side: 2, depthWrite: m.opacity >= 1 });
      if (mat.color) mat.color.set(m.color);
      if (mat.emissive) mat.emissive.set(m.color);
      const mesh = new Mesh(g, mat); mesh.name = 'overlay_' + m.name;
      mesh.matrixAutoUpdate = false; mesh.matrix.copy(F);
      link7.add(mesh); vm.__overlay.push(mesh); attached++;
    }
  }
  return 'ok: ' + vm.__overlay.length + ' meshes (' + attached + ' on the gripper)';
}
"""

# Makes the 3D view fill the page (covering the rest of the Studio UI) and sets the camera.
_JS_ARM_ONLY = r"""
(view) => {
  let vm = window.__arm3dVm || null;  // cached: the canvas is later moved out of the component
  for (let el = document.querySelector('canvas'); !vm && el; el = el.parentElement) { if (el.__vue__ && el.__vue__.$options.name === 'Arm3D') vm = el.__vue__; }
  window.__arm3dVm = vm;
  if (!vm || !vm.three || !vm.three.renderer) return 'Arm3D scene not found';
  const canvas = vm.three.renderer.domElement;
  if (!canvas) return 'renderer canvas not found';
  // Re-parent the canvas to <body> as a fixed full-page layer (keeps the WebGL context).
  document.body.appendChild(canvas);
  Object.assign(canvas.style, { position: 'fixed', left: '0', top: '0', width: '100vw', height: '100vh',
    zIndex: '2147483646', background: view.background });
  document.body.style.background = view.background;
  vm.size = null;
  const w = window.innerWidth, h = window.innerHeight;
  vm.three.renderer.setSize(w, h);
  const cam = vm.three.camera, ctl = vm.three.controls;
  cam.left = -w / 50; cam.right = w / 50; cam.top = h / 50; cam.bottom = -h / 50;
  // three.js coords of the target point (robot base frame mm -> scene)
  const tgt = [-view.target[0] / 100, view.target[2] / 100 - 5, view.target[1] / 100];
  const az = view.azimuth_deg * Math.PI / 180, el_ = view.elevation_deg * Math.PI / 180, r = 20;
  // azimuth measured in the robot base frame from +X toward +Y
  const d = [Math.cos(el_) * Math.cos(az), Math.sin(el_), Math.cos(el_) * Math.sin(az)];
  if (ctl) {
    ['minPolarAngle','maxPolarAngle','minAzimuthAngle','maxAzimuthAngle'].forEach(k => { if (k in ctl) ctl[k] = k.startsWith('min') ? -Infinity : Infinity; });
    ctl.minPolarAngle = 0; ctl.maxPolarAngle = Math.PI;
    ctl.target.set(...tgt);
  }
  cam.position.set(tgt[0] - d[0] * r, tgt[1] + d[1] * r, tgt[2] + d[2] * r);
  cam.lookAt(...tgt);
  cam.zoom = view.zoom;
  cam.updateProjectionMatrix();
  if (ctl) ctl.update();
  return 'ok';
}
"""


def install_overlay(page, config_path: str | Path | None = None, show_margin: bool | None = None, timeout_s: float = 15.0) -> str:
    spec = build_overlay_spec(config_path, show_margin)
    deadline = time.time() + timeout_s
    while True:
        result = page.evaluate(_JS_ADD_MESHES, spec)
        if result.startswith("ok") or time.time() > deadline:
            return result
        time.sleep(0.5)


def arm_only_view(
    page,
    azimuth_deg: float = 35.0,
    elevation_deg: float = 25.0,
    zoom: float | None = None,
    target_mm: tuple[float, float, float] | None = None,
    config_path: str | Path | None = None,
    background: str = "#1e1f26",
) -> str:
    """Fill the page with the 3D view. Default zoom/target frame the workcell box."""
    cfg = load_safety_config(config_path)
    if target_mm is None:
        # workcell center expressed in the base frame
        yaw = math.radians(cfg.base_yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        d = np.array([-cfg.base_x_mm, -cfg.base_y_mm, cfg.height_H_mm / 2 - cfg.base_z_mm])
        target_mm = tuple((np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) @ d).tolist())
    if zoom is None:
        vp = page.viewport_size or {"width": 1600, "height": 900}
        extent_units = max(cfg.depth_D_mm, cfg.width_W_mm, cfg.height_H_mm) / 100.0
        # ortho half-height is (height/50)/zoom units; make the largest box extent span ~80% of the frame
        zoom = 2 * 0.8 * min(vp["width"], vp["height"]) / 50.0 / extent_units
    view = {
        "azimuth_deg": azimuth_deg,
        "elevation_deg": elevation_deg,
        "zoom": zoom,
        "target": list(target_mm),
        "background": background,
    }
    return page.evaluate(_JS_ARM_ONLY, view)


def main(argv=None) -> int:
    """Live view: open Studio in a visible browser with the walls and the gripper model drawn in.
    Works against the simulator or the real robot's Studio (http://<robot ip>:18333)."""
    import argparse

    from playwright.sync_api import sync_playwright

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--url", default="http://127.0.0.1:18333")
    p.add_argument("--config", default=None, help="safety YAML (default config/safety.yaml)")
    p.add_argument("--arm-only", action="store_true", help="fill the window with the 3D view")
    p.add_argument("--seconds", type=float, default=0.0, help="close after this long (0 = until Ctrl+C)")
    p.add_argument("--screenshot", default=None, help="save a screenshot after setting up")
    p.add_argument("--headless", action="store_true")
    args = p.parse_args(argv)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=args.headless)
        page = browser.new_page(viewport={"width": 1400, "height": 850})
        page.goto(args.url, wait_until="domcontentloaded")
        try:
            page.get_by_role("button", name="Close").first.click(timeout=3000)
        except Exception:
            pass
        time.sleep(3)
        print("walls overlay:", install_overlay(page, args.config))
        if args.arm_only:
            print("arm-only view:", arm_only_view(page, config_path=args.config))
        if args.screenshot:
            time.sleep(1)
            page.screenshot(path=args.screenshot)
        print("Live view open; the drawn objects follow the arm. Ctrl+C to close.")
        try:
            t0 = time.time()
            while not args.seconds or time.time() - t0 < args.seconds:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        browser.close()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

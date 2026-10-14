"""Preprocessed sets: matching by setup fingerprint, set creation, clean-copy reuse and refits."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.xarm_pipeline.safety import fit, preprocessed as pp

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "safety_test.yaml"


def _write_cfg(path: Path, **changes) -> Path:
    raw = yaml.safe_load(FIXTURE.read_text())
    raw.setdefault("fit", {})["max_evals"] = 12  # keep the searches short
    for dotted, value in changes.items():
        node = raw
        *head, last = dotted.split(".")
        for k in head:
            node = node.setdefault(k, {})
        node[last] = value
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def _circle_csv(path: Path, z: float = 250.0, grouped: bool = False) -> Path:
    t = np.arange(0.0, 2.0 + 1e-9, 0.05)
    df = pd.DataFrame({"t": t, "x": 300 + 30 * np.sin(t), "y": 30 * np.cos(t), "z": z,
                       "roll": np.pi, "pitch": 0.0, "yaw": 0.0})
    if grouped:
        df = pd.concat([df.assign(traj="a"), df.assign(traj="b", z=z + 50)], ignore_index=True)
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def root(tmp_path, monkeypatch):
    r = tmp_path / "preprocessed"
    monkeypatch.setattr(pp, "PREPROCESSED_ROOT", r)
    return r


def test_fingerprint_ignores_non_setup_settings(tmp_path):
    base = pp.setup_fingerprint(_write_cfg(tmp_path / "a.yaml"))
    assert pp.setup_fingerprint(_write_cfg(tmp_path / "b.yaml", **{"runtime.latency_s": 0.5})) == base
    assert pp.setup_fingerprint(_write_cfg(tmp_path / "c.yaml", **{"visualization.wall_opacity": 0.3})) == base
    assert pp.setup_fingerprint(_write_cfg(tmp_path / "d.yaml", **{"firmware_fence.enabled": False})) == base
    commented = tmp_path / "e.yaml"
    commented.write_text("# a new comment\n" + (tmp_path / "a.yaml").read_text())
    assert pp.setup_fingerprint(commented) == base
    # 1200 vs 1200.0 is the same setup
    assert pp.setup_fingerprint(_write_cfg(tmp_path / "f.yaml", **{"workcell.depth_D_mm": 1200})) == base
    assert pp.setup_fingerprint(_write_cfg(tmp_path / "g.yaml", **{"workcell.depth_D_mm": 1000.0})) != base
    assert pp.setup_fingerprint(_write_cfg(tmp_path / "h.yaml", **{"fit.yaw_range_deg": 0})) != base
    assert pp.setup_fingerprint(_write_cfg(tmp_path / "i.yaml", **{"geometry.tool_offset_mm": [0, 0, 0]})) != base


def test_sets_are_created_once_per_setup(tmp_path, root):
    a = _write_cfg(tmp_path / "a.yaml")
    d1 = pp.find_or_create_set(a)
    assert d1.name == "001_D1200xW800xH1100"
    assert (d1 / "safety.yaml").read_text() == a.read_text()
    assert pp.find_or_create_set(_write_cfg(tmp_path / "b.yaml", **{"runtime.latency_s": 0.5})) == d1
    d2 = pp.find_or_create_set(_write_cfg(tmp_path / "c.yaml", **{"workcell.width_W_mm": 900.0}))
    assert d2.name == "002_D1200xW900xH1100"
    assert pp.find_set(a) == d1


def test_clean_path_mirrors_data_dir(root):
    d = root / "001_x"
    assert pp.clean_path_for(d, pp.REPO_ROOT / "data" / "trajectories" / "x.csv") == d / "trajectories" / "x.csv"
    ext = pp.clean_path_for(d, Path("/somewhere/else/y.csv"))
    assert ext.parent == d / "external" and ext.name.startswith("y_") and ext.suffix == ".csv"


def test_resolve_fits_once_then_reuses_and_refits_on_change(tmp_path, root, monkeypatch):
    cfg = _write_cfg(tmp_path / "a.yaml")
    src = _circle_csv(tmp_path / "circle.csv", z=60.0)  # gripper through the floor as given
    refs = pp.resolve(str(src), cfg)
    assert len(refs) == 1 and refs[0].startswith(str(root))
    clean = pd.read_csv(refs[0])
    assert clean["z"].min() > 60.0  # lifted
    assert np.allclose(np.diff(clean["x"]), np.diff(pd.read_csv(src)["x"]), atol=1e-5)  # relative motion kept

    calls = []
    real_fit = fit.fit_file
    monkeypatch.setattr(fit, "fit_file", lambda *a, **k: calls.append(a) or real_fit(*a, **k))
    assert pp.resolve(str(src), cfg) == refs and not calls  # up to date: reused

    _circle_csv(src, z=80.0)  # source edited
    assert pp.resolve(str(src), cfg) == refs and len(calls) == 1
    assert pp.resolve(refs[0], cfg) == [refs[0]]  # a clean ref resolves to itself

    other = _write_cfg(tmp_path / "b.yaml", **{"workcell.width_W_mm": 900.0})
    with pytest.raises(ValueError, match="another safety setup"):
        pp.resolve(refs[0], other)


def test_resolve_grouped_file_selects_and_merges(tmp_path, root):
    cfg = _write_cfg(tmp_path / "a.yaml")
    src = _circle_csv(tmp_path / "bank.csv", grouped=True)
    with pytest.raises(ValueError, match="select one"):
        pp.resolve(str(src), cfg)
    ra = pp.resolve(f"{src}#a", cfg)
    rb = pp.resolve(f"{src}#1", cfg)  # by index, fitted into the same clean bank file
    assert ra[0].endswith("#a") and rb[0].endswith("#b")
    assert ra[0].split("#")[0] == rb[0].split("#")[0]
    clean = pd.read_csv(ra[0].split("#")[0])
    assert list(pd.unique(clean["traj"])) == ["a", "b"]


def test_resolve_passes_joint_trajectories_through(tmp_path, root):
    t = np.arange(0.0, 1.0, 0.1)
    df = pd.DataFrame({"t": t, **{f"q{i}": 0.1 * i + 0 * t for i in range(1, 8)}})
    src = tmp_path / "joint.csv"
    df.to_csv(src, index=False)
    assert pp.resolve(str(src), _write_cfg(tmp_path / "a.yaml")) == [str(src)]
    assert not root.exists()  # nothing created for joint trajectories

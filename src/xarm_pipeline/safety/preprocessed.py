"""Preprocessed trajectory sets: one directory per safety setup (box, base pose, tool, fit options).

    data/preprocessed/
      001_D1200xW800xH1100/
        safety.yaml          frozen copy of the safety config this set was made with
        meta.json            index, fingerprint, creation time, source config path, box summary
        fit_log.jsonl        placements found in this set (seeds later searches)
        trajectories/bank.csv, bank.fit.json     clean copy of data/trajectories/bank.csv
      002_D600xW500xH1100/
        ...

Sets are matched by a fingerprint of the settings that change what a clean trajectory looks like
(workcell, base pose, margin, tool and link geometry, firmware TCP offset, IK/interpolation tolerances
and the 'fit:' section), not by comparing files byte for byte. Edits to comments, runtime monitor
timing, firmware-fence or visualization settings therefore reuse the existing set.

A clean copy stores the SHA-1 of its source file, so editing the source triggers a refit.

``resolve(ref)`` is what playback calls: it returns the clean trajectory reference(s) to execute,
fitting first when no up-to-date clean copy exists for the current config.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from ..trajectory_io import TRAJ_COLUMN, list_trajectories, split_trajectory_ref
from .config import DEFAULT_CONFIG_PATH

REPO_ROOT = Path(__file__).resolve().parents[3]
PREPROCESSED_ROOT = REPO_ROOT / "data" / "preprocessed"

# Top-level config keys (or key paths) that define a setup. Anything else (runtime, firmware_fence,
# visualization, preprocess.policy/enabled/clip_smoothing_s, comments) does not.
FINGERPRINT_KEYS = (
    ("workcell",),
    ("robot_base_in_world",),
    ("safety_margin_mm",),
    ("geometry",),
    ("preprocess", "interp_resolution_mm"),
    ("preprocess", "interp_resolution_deg"),
    ("preprocess", "ik_model_tolerance_mm"),
    ("fit",),
)


def _norm(v):
    """Make 1 and 1.0 (and nested lists/dicts of them) fingerprint the same."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        return {str(k): _norm(x) for k, x in sorted(v.items())}
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    return str(v)


def setup_subset(raw: dict) -> dict:
    out: dict = {}
    for path in FINGERPRINT_KEYS:
        node = raw
        for k in path:
            node = node.get(k) if isinstance(node, dict) else None
        out[".".join(path)] = _norm(node)
    return out


def setup_fingerprint(cfg_path: str | Path | None = None) -> str:
    from .config import load_safety_config

    raw = yaml.safe_load(Path(cfg_path or DEFAULT_CONFIG_PATH).read_text(encoding="utf-8")) or {}
    subset = setup_subset(raw)
    # The workcell can be written as a box + base pose or as walls in the base frame: fingerprint the
    # resulting geometry so both spellings of the same cell match.
    cfg = load_safety_config(cfg_path)
    subset["workcell"] = _norm([round(v, 3) + 0.0 for v in (cfg.depth_D_mm, cfg.width_W_mm, cfg.height_H_mm)])
    subset["robot_base_in_world"] = _norm([round(v, 3) + 0.0 for v in (cfg.base_x_mm, cfg.base_y_mm, cfg.base_z_mm,
                                                                 cfg.base_yaw_deg)])
    return hashlib.sha1(json.dumps(subset, sort_keys=True).encode()).hexdigest()[:12]


def list_sets(root: Path | None = None) -> list[tuple[Path, dict]]:
    root = root or PREPROCESSED_ROOT
    sets = []
    for d in sorted(root.glob("[0-9][0-9][0-9]_*")):
        meta = d / "meta.json"
        if meta.exists():
            sets.append((d, json.loads(meta.read_text(encoding="utf-8"))))
    return sets


def find_set(cfg_path: str | Path | None = None, root: Path | None = None) -> Path | None:
    root = root or PREPROCESSED_ROOT
    fp = setup_fingerprint(cfg_path)
    for d, meta in list_sets(root):
        if meta.get("fingerprint") == fp:
            return d
    return None


def find_or_create_set(cfg_path: str | Path | None = None, root: Path | None = None) -> Path:
    root = root or PREPROCESSED_ROOT
    d = find_set(cfg_path, root)
    if d is not None:
        return d
    cfg_file = Path(cfg_path or DEFAULT_CONFIG_PATH)
    raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
    from .config import load_safety_config

    cfg = load_safety_config(cfg_file)  # either workcell format
    label = f"D{cfg.depth_D_mm:g}xW{cfg.width_W_mm:g}xH{cfg.height_H_mm:g}"
    index = max([int(p.name[:3]) for p, _ in list_sets(root)] + [0]) + 1
    d = root / f"{index:03d}_{label}"
    d.mkdir(parents=True)
    shutil.copy2(cfg_file, d / "safety.yaml")
    meta = {
        "index": index,
        "fingerprint": setup_fingerprint(cfg_file),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_config": str(cfg_file),
        "box_D_W_H_mm": [cfg.depth_D_mm, cfg.width_W_mm, cfg.height_H_mm],
        "base_x_y_z_mm_yaw_deg": [cfg.base_x_mm, cfg.base_y_mm, cfg.base_z_mm, cfg.base_yaw_deg],
        "walls_in_base_mm": (raw.get("workcell") or {}).get("walls_in_base_mm"),
        "tool_offset_mm": (raw.get("geometry") or {}).get("tool_offset_mm"),
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"created preprocessed set {d} for config {cfg_file} (fingerprint {meta['fingerprint']})", file=sys.stderr)
    return d


def clean_path_for(set_dir: Path, src: Path) -> Path:
    """Mirror the source location inside the set: data/trajectories/x.csv -> <set>/trajectories/x.csv.
    Files outside data/ (e.g. another repo) go to <set>/external/<stem>_<path hash>.csv."""
    src = src.resolve()
    data_dir = REPO_ROOT / "data"
    try:
        return set_dir / src.relative_to(data_dir)
    except ValueError:
        pass
    try:
        return set_dir / "repo" / src.relative_to(REPO_ROOT)
    except ValueError:
        h = hashlib.sha1(str(src).encode()).hexdigest()[:8]
        return set_dir / "external" / f"{src.stem}_{h}{src.suffix}"


def set_of(path: Path, root: Path | None = None) -> Path | None:
    root = root or PREPROCESSED_ROOT
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return root / rel.parts[0] if rel.parts else None


def _is_joint_file(df: pd.DataFrame) -> bool:
    return all(f"q{i}" in df.columns for i in range(1, 8)) and "x" not in df.columns


def _refs_from_meta(meta: dict, out_path: Path, names: list[str]) -> list[str]:
    by_name = {e["name"]: e for e in meta["trajectories"]}
    refs = []
    for n in names:
        e = by_name[n]
        if not e["parts"]:
            raise ValueError(f"trajectory '{n}' could not be fitted into this workcell "
                             f"({e.get('reason') or e.get('result')}); see {out_path.with_suffix('.fit.json')}")
        if e.get("dropped"):
            print(f"WARNING: '{n}': {len(e['dropped'])} part(s) could not be fitted and are skipped: "
                  + ", ".join(f"t={d['source_t'][0]:.2f}-{d['source_t'][1]:.2f}s" for d in e["dropped"]), file=sys.stderr)
        for p in e["parts"]:
            refs.append(f"{out_path}#{p['name']}" if meta.get("grouped") else str(out_path))
    return refs


def start_joints_for(ref: str) -> list[float] | None:
    """Validated start joints (deg) stored by the fit for a clean trajectory ref, else None."""
    part = _fit_part_for(ref)
    return part.get("start_joints_deg") if part else None


def approach_for(ref: str) -> tuple[list[float], list[float]] | None:
    """(via joints, start joints) in deg for the validated approach of a clean trajectory ref:
    joint move to 'via' (the fit's seed / ready configuration), then joint move to 'start'."""
    part = _fit_part_for(ref)
    if not part or "start_joints_deg" not in part or "approach_via_joints_deg" not in part:
        return None
    return part["approach_via_joints_deg"], part["start_joints_deg"]


def _fit_part_for(ref: str) -> dict | None:
    src, selector = split_trajectory_ref(ref)
    meta_path = src.with_suffix(".fit.json")
    if set_of(src) is None or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    parts = [p for e in meta["trajectories"] for p in e["parts"]]
    for p in parts:
        if selector is None or p["name"] == selector:
            return p
    if selector is not None and selector.isdigit() and int(selector) < len(parts):
        return parts[int(selector)]
    return None


def resolve(ref: str, cfg_path: str | Path | None = None, refit: bool = False) -> list[str]:
    """Clean trajectory reference(s) to execute for ``ref`` under the current safety config.

    - ref already inside data/preprocessed/: returned as-is if its set matches the current config.
    - joint trajectories: returned as-is (they cannot be re-placed).
    - otherwise: the clean copy in the matching set (created / refitted when missing or stale).
      A trajectory that was split resolves to several parts, in execution order.
    """
    from .fit import FIT_VERSION, file_sha1, fit_file

    src, selector = split_trajectory_ref(ref)
    if not src.exists():
        raise FileNotFoundError(src)
    in_set = set_of(src)
    if in_set is not None:
        meta = json.loads((in_set / "meta.json").read_text(encoding="utf-8"))
        if meta["fingerprint"] != setup_fingerprint(cfg_path):
            raise ValueError(
                f"{ref} was preprocessed for another safety setup ({in_set.name}); "
                "run the original trajectory so it is fitted for the current config, or use that setup's config")
        return [ref]

    df = pd.read_csv(src, dtype={TRAJ_COLUMN: str})
    if _is_joint_file(df):
        return [ref]
    if TRAJ_COLUMN in df.columns:
        all_names = [n for n, _ in list_trajectories(df)]
        if selector is None and len(all_names) > 1:
            raise ValueError(f"{src} holds {len(all_names)} trajectories; select one with '#name': {all_names}")
        name = selector if selector in all_names else (all_names[int(selector)] if selector is not None else all_names[0])
        if any(f"q{i}" in df.columns for i in range(1, 8)):  # mixed file: is this one a joint trajectory?
            part = df[df[TRAJ_COLUMN] == name].dropna(axis=1, how="all")
            if _is_joint_file(part):
                return [ref]
    else:
        name = src.stem

    set_dir = find_or_create_set(cfg_path)
    out_path = clean_path_for(set_dir, src)
    meta_path = out_path.with_suffix(".fit.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    fresh = (meta is not None and meta.get("source_sha1") == file_sha1(src) and meta.get("fit_version") == FIT_VERSION
             and any(e["name"] == name and all("approach_via_joints_deg" in p for p in e["parts"])
                     for e in meta["trajectories"]))  # older clean copies lack start joints: refit
    if refit or not fresh:
        if refit:
            why = "forced refit"
        elif meta is None or not any(e["name"] == name for e in meta["trajectories"]):
            why = "no clean copy yet"
        elif meta.get("source_sha1") != file_sha1(src):
            why = "source file changed"
        elif meta.get("fit_version") != FIT_VERSION:
            why = "made by an older fitting algorithm"
        else:
            why = "clean copy predates the stored approach data"
        print(f"fitting {src}#{name} for set {set_dir.name} ({why}); this can take a minute...", file=sys.stderr)
        sel_ref = f"{src}#{name}" if TRAJ_COLUMN in df.columns else str(src)
        meta = fit_file(sel_ref, out_path, cfg_path, set_dir / "fit_log.jsonl")
    else:
        print(f"using clean copy from set {set_dir.name}: {out_path}", file=sys.stderr)
    return _refs_from_meta(meta, out_path, [name])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Preprocessed trajectory sets (one per safety setup)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("resolve", help="print the clean trajectory ref(s) to execute, fitting if needed")
    r.add_argument("trajectory")
    r.add_argument("--config", default=None)
    r.add_argument("--refit", action="store_true", help="refit even if an up-to-date clean copy exists")
    sub.add_parser("list", help="list sets and their contents")
    s = sub.add_parser("which", help="print the set directory matching a config (creating none)")
    s.add_argument("--config", default=None)
    args = p.parse_args(argv)

    if args.cmd == "resolve":
        # stdout carries only the refs (scripts read it); fitting progress goes to stderr
        try:
            with contextlib.redirect_stdout(sys.stderr):
                refs = resolve(args.trajectory, args.config, args.refit)
            for x in refs:
                print(x)
        except (ValueError, FileNotFoundError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        return 0
    if args.cmd == "which":
        d = find_set(args.config)
        print(d if d else f"(no set yet for fingerprint {setup_fingerprint(args.config)})")
        return 0
    current = setup_fingerprint(None)
    for d, meta in list_sets():
        mark = "  <- current config/safety.yaml" if meta["fingerprint"] == current else ""
        print(f"{d.name}: fingerprint {meta['fingerprint']}, created {meta['created']}, box {meta['box_D_W_H_mm']}, "
              f"base {meta['base_x_y_z_mm_yaw_deg']}, tool {meta['tool_offset_mm']}{mark}")
        for f in sorted(d.rglob("*.fit.json")):
            m = json.loads(f.read_text(encoding="utf-8"))
            for e in m["trajectories"]:
                print(f"    {f.relative_to(d).with_suffix('').with_suffix('.csv')}#{e['name']}: {e['result']}, "
                      f"{len(e['parts'])} part(s)" + (f", {len(e['dropped'])} dropped" if e.get("dropped") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

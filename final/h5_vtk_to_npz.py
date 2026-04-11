"""Combine H5 solver data + VTK geometry references into unified NPZ frames.

Paths are configured externally; this script does not assume fixed folder layout.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .io import ensure_dir, load_config, save_json


FRAME_ID_REGEX = re.compile(r"(\d+)(?!.*\d)")


def parse_frame_id(path: Path) -> str:
    m = FRAME_ID_REGEX.search(path.stem)
    return m.group(1).zfill(6) if m else path.stem


def discover_many(patterns: Sequence[str], root: Path) -> List[Path]:
    out: List[Path] = []
    for pat in patterns:
        out.extend(sorted(root.glob(pat)))
    return sorted(set(out))


def _h5_read_dataset(h5f, dataset_path: str) -> np.ndarray:
    if dataset_path not in h5f:
        raise KeyError(f"Dataset path not found in H5: {dataset_path}")
    return np.asarray(h5f[dataset_path])


def _load_h5_map(h5_path: Path, key_map: Mapping[str, str]) -> Dict[str, np.ndarray]:
    try:
        import h5py  # type: ignore
    except Exception as exc:
        raise RuntimeError("h5py is required for H5->NPZ conversion") from exc

    out: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as h5f:
        for out_key, in_path in key_map.items():
            if not in_path:
                continue
            out[out_key] = _h5_read_dataset(h5f, in_path)
    return out


def _pair_by_frame(h5_files: Sequence[Path], vtk_files: Sequence[Path]) -> List[Tuple[Path, Optional[Path], str]]:
    vtk_map = {parse_frame_id(v): v for v in vtk_files}
    pairs = []
    for h5 in sorted(h5_files):
        fid = parse_frame_id(h5)
        pairs.append((h5, vtk_map.get(fid), fid))
    return pairs


def convert_h5_vtk_to_npz(cfg: Mapping[str, Any], root: Path) -> Dict[str, Any]:
    io_cfg = cfg.get("io", {})
    h5_patterns = list(io_cfg.get("h5_glob", []))
    vtk_patterns = list(io_cfg.get("vtk_glob", []))

    if not h5_patterns:
        raise RuntimeError("Config io.h5_glob is empty. Fill H5 path patterns first.")

    h5_files = discover_many(h5_patterns, root)
    vtk_files = discover_many(vtk_patterns, root) if vtk_patterns else []

    key_map = dict(cfg.get("key_map", {}))
    if not key_map:
        raise RuntimeError("Config key_map is empty. Map output npz keys to H5 dataset paths.")

    out_dir = ensure_dir(root / cfg.get("output_dir", "final/output/merged_npz"))

    pairs = _pair_by_frame(h5_files, vtk_files)
    written = 0
    missing_vtk = 0
    records = []

    for h5_path, vtk_path, fid in pairs:
        try:
            payload = _load_h5_map(h5_path, key_map)
        except Exception as exc:
            print(f"[warn] skip {h5_path}: {exc}")
            continue

        if vtk_path is None:
            missing_vtk += 1
            vtk_str = ""
        else:
            vtk_str = str(vtk_path)

        payload["source_h5_path"] = np.asarray(str(h5_path), dtype=object)
        payload["source_vtk_path"] = np.asarray(vtk_str, dtype=object)

        out_path = out_dir / f"frame_{fid}.npz"
        np.savez_compressed(out_path, **payload)
        written += 1

        records.append(
            {
                "frame_id": fid,
                "h5_path": str(h5_path),
                "vtk_path": vtk_str,
                "npz_path": str(out_path),
                "keys": sorted(payload.keys()),
            }
        )

    summary = {
        "n_h5": len(h5_files),
        "n_vtk": len(vtk_files),
        "n_written": written,
        "missing_vtk": missing_vtk,
        "output_dir": str(out_dir),
        "key_map": key_map,
    }
    save_json(out_dir / "conversion_summary.json", summary)
    save_json(out_dir / "conversion_index.json", {"frames": records})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert H5 + VTK references into unified NPZ frames")
    parser.add_argument("--config", type=str, default="final/configs/h5_vtk_to_npz_template.yaml")
    parser.add_argument("--root", type=str, default=".")
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary = convert_h5_vtk_to_npz(cfg, root=Path(args.root))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

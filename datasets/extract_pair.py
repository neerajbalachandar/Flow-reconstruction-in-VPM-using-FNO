import os
import json
import numpy as np
import h5py
from pathlib import Path
from scipy.spatial import cKDTree

# ============================================================
# PATHS
# ============================================================

pfield_folder = Path("")
fdom_folder   = Path("")

pfield_basename = ""
fdom_basename   = ""

out_folder = Path("training_pairs_clean")
out_folder.mkdir(parents=True, exist_ok=True)

# ============================================================
# FRAME RANGE
# ============================================================

FRAME_START = 100
FRAME_STOP  = 400
FRAME_STEP  = 10

# ============================================================
# READ PFIELD
# ============================================================

def read_required_pfield(pfile: Path):
    with h5py.File(pfile, "r") as hf:
        needed = ["X", "Gamma", "sigma", "circulation", "vol", "static", "i", "C"]
        for k in needed:
            if k not in hf:
                raise KeyError(f"Dataset '{k}' not found in {pfile}")

        pos = np.array(hf["X"])
        Gamma_vec = np.array(hf["Gamma"])
        sigma = np.array(hf["sigma"])
        circulation = np.array(hf["circulation"])
        vol = np.array(hf["vol"])
        static = np.array(hf["static"])
        idx_i = np.array(hf["i"])
        C = np.array(hf["C"])

    N = pos.shape[0]

    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"pos must be (N,3), got {pos.shape}")

    if Gamma_vec.ndim == 1 and Gamma_vec.size == 3 * N:
        Gamma_vec = Gamma_vec.reshape(N, 3)

    if Gamma_vec.ndim != 2 or Gamma_vec.shape[0] != N:
        raise ValueError("Gamma shape mismatch")

    gamma_scalar = np.linalg.norm(Gamma_vec, axis=1)

    return {
        "pos": pos.astype(np.float64),
        "Gamma_vec": Gamma_vec.astype(np.float64),
        "gamma_scalar": gamma_scalar.astype(np.float64),
        "sigma": sigma.astype(np.float64),
        "circulation": circulation.astype(np.float64),
        "vol": vol.astype(np.float64),
        "static": static.astype(np.int32),
        "i": idx_i.astype(np.int32),
        "C": C.astype(np.float64),
    }

# ============================================================
# READ FDOM
# ============================================================

def read_required_fdom(ffile: Path):
    with h5py.File(ffile, "r") as hf:
        if "nodes" in hf:
            nodes = np.array(hf["nodes"])
        elif "X" in hf:
            nodes = np.array(hf["X"])
        else:
            raise KeyError("'nodes' or 'X' not found")

        if "U" not in hf:
            raise KeyError("'U' not found")

        U_arr = np.array(hf["U"])

    if nodes.ndim == 2 and nodes.shape[1] >= 2:
        nodes = nodes[:, :2]
    else:
        raise ValueError("Unsupported nodes shape")

    if U_arr.ndim == 2:
        U_nodes = U_arr[:, :2]
    elif U_arr.ndim == 4:
        U_nodes = U_arr.reshape(-1, U_arr.shape[-1])[:, :2]
    else:
        raise ValueError("Unsupported U shape")

    if nodes.shape[0] != U_nodes.shape[0]:
        raise ValueError("nodes and U_nodes size mismatch")

    return {"nodes": nodes, "U_nodes": U_nodes}

# ============================================================
# SAMPLE VELOCITY AT PARTICLES
# ============================================================

def sample_velocity_at_particles(nodes, U_nodes, particle_pos):
    tree = cKDTree(nodes)
    _, idx = tree.query(particle_pos[:, :2], k=1)
    return U_nodes[idx]

# ============================================================
# MAIN LOOP
# ============================================================

frames = list(range(FRAME_START, FRAME_STOP + 1, FRAME_STEP))
metadata = {"frames": []}

for frame in frames:
    ppath = pfield_folder / f"{pfield_basename}.{frame}.h5"
    fpath = fdom_folder / f"{fdom_basename}.{frame}.h5"

    if not ppath.exists():
        print(f"[skip] missing {ppath}")
        continue
    if not fpath.exists():
        print(f"[skip] missing {fpath}")
        continue

    try:
        pdat = read_required_pfield(ppath)
        fdat = read_required_fdom(fpath)

        U_at_particles = sample_velocity_at_particles(
            fdat["nodes"], fdat["U_nodes"], pdat["pos"]
        )

        out = {
            **pdat,
            "nodes": fdat["nodes"],
            "U_nodes": fdat["U_nodes"],
            "U_at_particles": U_at_particles,
            "frame": np.int32(frame),
            "source_files": np.array([str(ppath), str(fpath)], dtype=object),
        }

        outname = out_folder / f"frame_{frame}.npz"
        np.savez_compressed(outname, **out)

        metadata["frames"].append({
            "frame": int(frame),
            "n_particles": int(pdat["pos"].shape[0]),
            "n_nodes": int(fdat["nodes"].shape[0]),
            "file": str(outname)
        })

        print(f"[saved] {outname}")

    except Exception as e:
        print(f"[error frame {frame}] {e}")

# ============================================================
# WRITE METADATA
# ============================================================

meta_path = out_folder / "metadata_frames.json"
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nDone. Metadata written to {meta_path}")

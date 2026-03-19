import os
import json
import numpy as np
import h5py
from scipy.spatial import cKDTree

# ---------------- paths & config ----------------
pfield_folder = "/home/dysco/Neeraj/Flow-reconstruction-in-VPM-using-FNO/data/raw data/poisson_dataset2_simplewing/input/"
fdom_folder   = "/home/dysco/Neeraj/Flow-reconstruction-in-VPM-using-FNO/data/raw data/poisson_dataset2_simplewing/output/"

pfield_basename = "simple-wing_pfield"
fdom_basename   = "simple-wing_fdom_fdom"   

OUT_FOLDER = "/home/dysco/Neeraj/Flow-reconstruction-in-VPM-using-FNO/data/train/pair_2/"

FRAME_START = 50
FRAME_STOP  = 190
FRAME_STEP  = 10

os.makedirs(OUT_FOLDER, exist_ok=True)

# ---------------- I/O helpers ----------------
def read_required_pfield(pfile):
    """Read FLOWVPM particle-field data with optional C and i."""
    with h5py.File(pfile, "r") as hf:
        required = ["X", "Gamma", "sigma", "circulation", "vol", "static"]
        for k in required:
            if k not in hf:
                raise KeyError(f"Dataset '{k}' not found in {pfile}")

        pos         = np.array(hf["X"])
        Gamma_vec   = np.array(hf["Gamma"])
        sigma       = np.array(hf["sigma"])
        circulation = np.array(hf["circulation"])
        vol         = np.array(hf["vol"])
        static      = np.array(hf["static"])

        # OPTIONAL datasets
        idx_i = np.array(hf["i"]) if "i" in hf else None
        C     = np.array(hf["C"]) if "C" in hf else None

    # ---- shape checks ----
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"X must be (N,3); got {pos.shape}")
    N = pos.shape[0]

    # Normalize Gamma
    if Gamma_vec.ndim == 1 and Gamma_vec.size == 3 * N:
        Gamma_vec = Gamma_vec.reshape(N, 3)
    elif Gamma_vec.ndim == 2 and Gamma_vec.shape == (3, N):
        Gamma_vec = Gamma_vec.T

    if Gamma_vec.ndim != 2 or Gamma_vec.shape[0] != N:
        raise ValueError(f"Invalid Gamma shape {Gamma_vec.shape}")

    # Handle optional i
    if idx_i is None:
        idx_i = np.arange(N, dtype=np.int32)
    else:
        if idx_i.shape[0] != N:
            raise ValueError("Dataset 'i' has wrong length")

    # Handle optional C  ✅ THIS IS THE KEY FIX
    if C is None:
        C = pos.copy()
    else:
        if C.ndim != 2 or C.shape[0] != N:
            raise ValueError("Dataset 'C' has wrong shape")

    # Validate remaining arrays
    for name, arr in [
        ("sigma", sigma),
        ("circulation", circulation),
        ("vol", vol),
        ("static", static),
    ]:
        if arr.shape[0] != N:
            raise ValueError(f"{name} must have length N={N}")

    gamma_mag = np.linalg.norm(Gamma_vec, axis=1)

    return {
        "pos": pos.astype(np.float64),
        "Gamma_vec": Gamma_vec.astype(np.float64),
        "gamma_mag": gamma_mag.astype(np.float64),
        "sigma": sigma.astype(np.float64),
        "circulation": circulation.astype(np.float64),
        "vol": vol.astype(np.float64),
        "static": static.astype(np.int32),
        "i": idx_i.astype(np.int32),
        "C": C.astype(np.float64),
    }



def read_required_fdom(ffile):
    """Read fluid-domain nodes and velocity."""
    with h5py.File(ffile, "r") as hf:
        nodes = np.array(hf["nodes"] if "nodes" in hf else hf["X"])
        U_arr = np.array(hf["U"])

    if nodes.ndim >= 3:
        nodes = nodes.reshape(-1, nodes.shape[-1])
    if U_arr.ndim >= 3:
        U_arr = U_arr.reshape(-1, U_arr.shape[-1])

    if nodes.shape[0] != U_arr.shape[0]:
        raise ValueError("nodes and U size mismatch")

    return {
        "nodes": nodes.astype(np.float64),
        "U_nodes": U_arr.astype(np.float64),
    }


def sample_velocity_at_particles(nodes, U_nodes, particle_pos):
    tree = cKDTree(nodes)
    _, idx = tree.query(particle_pos, k=1)
    return U_nodes[idx]


# ---------------- main processing ----------------
frames = list(range(FRAME_START, FRAME_STOP + 1, FRAME_STEP))
metadata = {"frames": []}

for frame in frames:
    ppath = os.path.join(pfield_folder, f"{pfield_basename}.{frame}.h5")
    fpath = os.path.join(fdom_folder,   f"{fdom_basename}.{frame}.h5")

    if not os.path.isfile(ppath) or not os.path.isfile(fpath):
        print(f"[skip] frame {frame} missing input")
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
            "source_files": np.array([ppath, fpath], dtype="U256"),
        }

        outname = os.path.join(OUT_FOLDER, f"frame_{frame}.npz")
        np.savez_compressed(outname, **out)

        metadata["frames"].append({
            "frame": frame,
            "n_particles": int(pdat["pos"].shape[0]),
            "n_nodes": int(fdat["nodes"].shape[0]),
            "file": outname,
        })

        print(f"[saved] frame {frame}")

    except Exception as e:
        print(f"[error frame {frame}] {e}")

# ---------------- metadata ----------------
meta_path = os.path.join(OUT_FOLDER, "metadata_frames.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nDone. Metadata written to {meta_path}")

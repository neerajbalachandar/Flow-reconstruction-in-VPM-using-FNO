import os
import json
import numpy as np
import h5py
from scipy.spatial import cKDTree

# ---------------- paths & config ----------------
pfield_folder = "/media/dysco/New Volume/Neeraj/neuralop/data/raw data/5/input/"
fdom_folder   = "/media/dysco/New Volume/Neeraj/neuralop/data/raw data/5/output/"

pfield_basename = "tethered-wing_pfield"
fdom_basename   = "tethered-wing_fdom"   

OUT_FOLDER = "/media/dysco/New Volume/Neeraj/neuralop/data/train/pair_5_gno/"

# Particle field (input) frame range/cadence
PFRAME_START = 0
PFRAME_STOP  = 360
PFRAME_STEP  = 1

# Fluid domain (output) frame range/cadence
FFRAME_START = 0
FFRAME_STOP  = 358
FFRAME_STEP  = 2

# Pairing mode:
# - "intersection": keep only frame ids present in both (same-time pairing)
# - "nearest_prev": for each pframe, use latest available fframe <= pframe
PAIR_MODE = "intersection"

os.makedirs(OUT_FOLDER, exist_ok=True)

# ---------------- I/O helpers ----------------
def ensure_n_by_3(arr, name):
    arr = np.asarray(arr)

    if arr.ndim == 1:
        if arr.size % 3 != 0:
            raise ValueError(f"{name} cannot be reshaped to (N,3); got shape {arr.shape}")
        arr = arr.reshape(-1, 3)
    elif arr.ndim == 2:
        if arr.shape[1] == 3:
            pass
        elif arr.shape[0] == 3:
            arr = arr.T
        else:
            raise ValueError(f"{name} must be (N,3) or (3,N); got {arr.shape}")
    else:
        # Handles structured-grid cases that come as (..., 3)
        arr = arr.reshape(-1, arr.shape[-1])
        if arr.shape[1] != 3:
            if arr.shape[0] == 3:
                arr = arr.T
            else:
                raise ValueError(f"{name} last dimension must be 3; got {arr.shape}")

    return np.ascontiguousarray(arr)


def ensure_vec_n(arr, n, name):
    arr = np.asarray(arr).squeeze()
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if arr.size != n:
        raise ValueError(f"{name} must have length N={n}; got {arr.shape}")
    return np.ascontiguousarray(arr)


def frame_dict(folder, basename, start, stop, step):
    frame_ids = list(range(start, stop + 1, step))
    return {
        fr: os.path.join(folder, f"{basename}.{fr}.h5")
        for fr in frame_ids
    }


def pair_frames(pfiles, ffiles, mode="intersection"):
    pkeys = sorted([k for k, v in pfiles.items() if os.path.isfile(v)])
    fkeys = sorted([k for k, v in ffiles.items() if os.path.isfile(v)])

    if mode == "intersection":
        pairs = [(fr, fr) for fr in pkeys if fr in ffiles and os.path.isfile(ffiles[fr])]
        return pairs

    if mode == "nearest_prev":
        pairs = []
        if len(fkeys) == 0:
            return pairs
        farr = np.array(fkeys, dtype=np.int64)
        for pfr in pkeys:
            idx = np.searchsorted(farr, pfr, side="right") - 1
            if idx < 0:
                continue
            ffr = int(farr[idx])
            pairs.append((pfr, ffr))
        return pairs

    raise ValueError("PAIR_MODE must be 'intersection' or 'nearest_prev'")


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
    pos = ensure_n_by_3(pos, "X")
    N = pos.shape[0]

    # Normalize Gamma
    if Gamma_vec.ndim == 1 and Gamma_vec.size == 3 * N:
        Gamma_vec = Gamma_vec.reshape(N, 3)
    elif Gamma_vec.ndim == 2 and Gamma_vec.shape == (3, N):
        Gamma_vec = Gamma_vec.T
    elif Gamma_vec.ndim == 2 and Gamma_vec.shape == (N, 3):
        pass
    else:
        Gamma_vec = ensure_n_by_3(Gamma_vec, "Gamma")

    if Gamma_vec.ndim != 2 or Gamma_vec.shape[0] != N or Gamma_vec.shape[1] != 3:
        raise ValueError(f"Invalid Gamma shape {Gamma_vec.shape}")

    # Handle optional i
    if idx_i is None:
        idx_i = np.arange(N, dtype=np.int32)
    else:
        idx_i = np.asarray(idx_i).reshape(-1)
        if idx_i.size != N:
            raise ValueError("Dataset 'i' has wrong length")

    # Handle optional C
    if C is None:
        C = pos.copy()
    else:
        C = ensure_n_by_3(C, "C")
        if C.shape[0] != N:
            raise ValueError("Dataset 'C' has wrong shape")

    # Validate remaining arrays
    sigma = ensure_vec_n(sigma, N, "sigma")
    circulation = ensure_vec_n(circulation, N, "circulation")
    vol = ensure_vec_n(vol, N, "vol")
    static = ensure_vec_n(static, N, "static")

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

    nodes = ensure_n_by_3(nodes, "nodes/X")
    U_arr = ensure_n_by_3(U_arr, "U")

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
pfiles = frame_dict(
    pfield_folder, pfield_basename, PFRAME_START, PFRAME_STOP, PFRAME_STEP
)
ffiles = frame_dict(
    fdom_folder, fdom_basename, FFRAME_START, FFRAME_STOP, FFRAME_STEP
)
pairs = pair_frames(pfiles, ffiles, mode=PAIR_MODE)

metadata = {"frames": []}

print(
    f"[info] Pair mode={PAIR_MODE}, particle_range=({PFRAME_START}:{PFRAME_STOP}:{PFRAME_STEP}), "
    f"fdom_range=({FFRAME_START}:{FFRAME_STOP}:{FFRAME_STEP}), n_pairs={len(pairs)}"
)

for pframe, fframe in pairs:
    ppath = pfiles[pframe]
    fpath = ffiles[fframe]

    if not os.path.isfile(ppath) or not os.path.isfile(fpath):
        print(f"[skip] pframe={pframe}, fframe={fframe} missing input")
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
            "frame": np.int32(pframe),
            "fdom_frame": np.int32(fframe),
            "source_files": np.array([ppath, fpath], dtype="U256"),
        }

        outname = os.path.join(OUT_FOLDER, f"frame_{pframe}.npz")
        np.savez_compressed(outname, **out)

        metadata["frames"].append({
            "frame": int(pframe),
            "fdom_frame": int(fframe),
            "n_particles": int(pdat["pos"].shape[0]),
            "n_nodes": int(fdat["nodes"].shape[0]),
            "file": outname,
        })

        print(f"[saved] frame={pframe} (fdom_frame={fframe})")

    except Exception as e:
        print(f"[error pframe={pframe}, fframe={fframe}] {e}")

# ---------------- metadata ----------------
meta_path = os.path.join(OUT_FOLDER, "metadata_frames.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nDone. Metadata written to {meta_path}")

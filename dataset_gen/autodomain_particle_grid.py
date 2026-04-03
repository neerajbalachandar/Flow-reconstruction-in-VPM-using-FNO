import argparse
import json
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def frame_id_from_path(path_obj):
    stem = path_obj.stem  # frame_<id> or frame_<id>_grid
    parts = stem.split("_")
    for tok in parts:
        if tok.isdigit():
            return int(tok)
    return 10**12


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert 3D particle/node frame data to 3D Eulerian grids with auto domain "
            "and multiple input channel variants (omega/no-omega/omega4)."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/media/dysco/New Volume/Neeraj/neuralop/data/train/pair_3_gno"),
        help="Folder containing frame_*.npz 3D particle files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/media/dysco/New Volume/Neeraj/neuralop/data/train/pair_3_fno_64"),
        help="Output folder for frame_*_grid.npz files.",
    )
    parser.add_argument("--nx", type=int, default=64, help="Grid resolution in x.")
    parser.add_argument("--ny", type=int, default=64, help="Grid resolution in y.")
    parser.add_argument("--nz", type=int, default=64, help="Grid resolution in z.")
    parser.add_argument(
        "--margin-frac",
        type=float,
        default=0.05,
        help="Fractional domain padding added to each side.",
    )
    parser.add_argument(
        "--kernel-trunc-sigma",
        type=float,
        default=3.0,
        help="Kernel support radius in units of sigma.",
    )
    parser.add_argument(
        "--sigma-floor",
        type=float,
        default=1e-6,
        help="Lower bound for sigma to avoid divide-by-zero.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="If > 0, process only first N frames (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--save-compressed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write output with np.savez_compressed (smaller files, slower CPU).",
    )
    return parser.parse_args()


def compute_auto_domain_3d(files, margin_frac):
    xmin, ymin, zmin = np.inf, np.inf, np.inf
    xmax, ymax, zmax = -np.inf, -np.inf, -np.inf

    for file_path in tqdm(files, desc="Scanning domain bounds", unit="frame"):
        data = np.load(file_path)

        pos_xyz = np.asarray(data["pos"][:, :3], dtype=np.float64)
        nodes_xyz = np.asarray(data["nodes"][:, :3], dtype=np.float64)
        all_xyz = np.concatenate([pos_xyz, nodes_xyz], axis=0)

        xmin = min(xmin, float(all_xyz[:, 0].min()))
        xmax = max(xmax, float(all_xyz[:, 0].max()))
        ymin = min(ymin, float(all_xyz[:, 1].min()))
        ymax = max(ymax, float(all_xyz[:, 1].max()))
        zmin = min(zmin, float(all_xyz[:, 2].min()))
        zmax = max(zmax, float(all_xyz[:, 2].max()))

    x_span = max(xmax - xmin, 1e-12)
    y_span = max(ymax - ymin, 1e-12)
    z_span = max(zmax - zmin, 1e-12)

    x_pad = margin_frac * x_span
    y_pad = margin_frac * y_span
    z_pad = margin_frac * z_span

    return (
        xmin - x_pad,
        xmax + x_pad,
        ymin - y_pad,
        ymax + y_pad,
        zmin - z_pad,
        zmax + z_pad,
    )


def in_domain_mask_3d(points_xyz, xmin, xmax, ymin, ymax, zmin, zmax):
    return (
        (points_xyz[:, 0] >= xmin)
        & (points_xyz[:, 0] <= xmax)
        & (points_xyz[:, 1] >= ymin)
        & (points_xyz[:, 1] <= ymax)
        & (points_xyz[:, 2] >= zmin)
        & (points_xyz[:, 2] <= zmax)
    )


def nearest_uniform_index(values, vmin, vmax, n):
    if n <= 1 or vmax <= vmin:
        return np.zeros_like(values, dtype=np.int64)
    scaled = (values - vmin) / (vmax - vmin)
    idx = np.rint(scaled * (n - 1)).astype(np.int64)
    return np.clip(idx, 0, n - 1)


def zeta_3d(q):
    # Normalized 3D Gaussian radial basis.
    return np.exp(-(q ** 2)) / (np.pi ** 1.5)


def project_particles_to_grid_3d(
    pos,
    gamma,
    sigma,
    vol,
    x_grid,
    y_grid,
    z_grid,
    X,
    Y,
    Z,
    bounds,
    kernel_trunc_sigma,
    sigma_floor,
):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    nx, ny, nz = X.shape
    omega_grid = np.zeros((nx, ny, nz), dtype=np.float32)
    gamma_grid = np.zeros((nx, ny, nz), dtype=np.float32)
    sigma_grid = np.zeros((nx, ny, nz), dtype=np.float32)
    pos_density = np.zeros((nx, ny, nz), dtype=np.float32)
    vol_grid = np.zeros((nx, ny, nz), dtype=np.float32)

    domain_mask = in_domain_mask_3d(pos, xmin, xmax, ymin, ymax, zmin, zmax)
    pos = pos[domain_mask]
    gamma = gamma[domain_mask]
    sigma = sigma[domain_mask]
    vol = vol[domain_mask]

    dropped_oob = int((~domain_mask).sum())
    dropped_bad_sigma = 0

    for p in range(pos.shape[0]):
        h = max(float(sigma[p]), sigma_floor)
        if not np.isfinite(h) or h <= 0.0:
            dropped_bad_sigma += 1
            continue

        radius = kernel_trunc_sigma * h
        px, py, pz = float(pos[p, 0]), float(pos[p, 1]), float(pos[p, 2])

        ix0 = np.searchsorted(x_grid, px - radius, side="left")
        ix1 = np.searchsorted(x_grid, px + radius, side="right")
        iy0 = np.searchsorted(y_grid, py - radius, side="left")
        iy1 = np.searchsorted(y_grid, py + radius, side="right")
        iz0 = np.searchsorted(z_grid, pz - radius, side="left")
        iz1 = np.searchsorted(z_grid, pz + radius, side="right")

        if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
            continue

        dx = X[ix0:ix1, iy0:iy1, iz0:iz1] - px
        dy = Y[ix0:ix1, iy0:iy1, iz0:iz1] - py
        dz = Z[ix0:ix1, iy0:iy1, iz0:iz1] - pz
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        q = r / h

        zeta_sigma = zeta_3d(q) / (h ** 3)
        omega_grid[ix0:ix1, iy0:iy1, iz0:iz1] += gamma[p] * zeta_sigma

        w = np.exp(-(q ** 2))
        gamma_grid[ix0:ix1, iy0:iy1, iz0:iz1] += gamma[p] * w
        sigma_grid[ix0:ix1, iy0:iy1, iz0:iz1] += sigma[p] * w
        pos_density[ix0:ix1, iy0:iy1, iz0:iz1] += w
        vol_grid[ix0:ix1, iy0:iy1, iz0:iz1] += vol[p] * w

    input_grid = np.stack([gamma_grid, sigma_grid, vol_grid], axis=0).astype(np.float32)
    input_grid_no_omega = np.stack([gamma_grid, sigma_grid, pos_density], axis=0).astype(np.float32)
    input_grid_omega = omega_grid[None, :, :, :].astype(np.float32)
    input_grid_omega4 = np.stack([omega_grid, gamma_grid, sigma_grid, pos_density], axis=0).astype(np.float32)

    return {
        "input_grid": input_grid,
        "input_grid_no_omega": input_grid_no_omega,
        "input_grid_omega": input_grid_omega,
        "input_grid_omega4": input_grid_omega4,
        "n_particles_used": int(pos.shape[0]),
        "n_particles_dropped_oob": dropped_oob,
        "n_particles_dropped_bad_sigma": dropped_bad_sigma,
    }


def velocity_to_grid_3d(nodes, U_nodes, x_grid, y_grid, z_grid, bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    nx, ny, nz = len(x_grid), len(y_grid), len(z_grid)
    u_grid = np.zeros((3, nx, ny, nz), dtype=np.float32)
    count = np.zeros((nx, ny, nz), dtype=np.float32)

    domain_mask = in_domain_mask_3d(nodes, xmin, xmax, ymin, ymax, zmin, zmax)
    dropped_oob = int((~domain_mask).sum())

    nodes = nodes[domain_mask]
    U_nodes = U_nodes[domain_mask]

    if nodes.shape[0] == 0:
        return u_grid, 0, dropped_oob

    ix = nearest_uniform_index(nodes[:, 0], x_grid[0], x_grid[-1], nx)
    iy = nearest_uniform_index(nodes[:, 1], y_grid[0], y_grid[-1], ny)
    iz = nearest_uniform_index(nodes[:, 2], z_grid[0], z_grid[-1], nz)

    np.add.at(u_grid[0], (ix, iy, iz), U_nodes[:, 0])
    np.add.at(u_grid[1], (ix, iy, iz), U_nodes[:, 1])
    np.add.at(u_grid[2], (ix, iy, iz), U_nodes[:, 2])
    np.add.at(count, (ix, iy, iz), 1.0)

    count[count == 0] = 1.0
    u_grid /= count

    return u_grid, int(nodes.shape[0]), dropped_oob


def main():
    args = parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("frame_*.npz"), key=frame_id_from_path)
    if not files:
        raise RuntimeError(f"No frame_*.npz files found in {data_dir}")

    if args.max_frames > 0:
        files = files[: args.max_frames]

    print(f"Input dir : {data_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Found {len(files)} frame files")
    if files:
        print(f"First file: {files[0].name}")
        print(f"Last file : {files[-1].name}")

    bounds = compute_auto_domain_3d(files, args.margin_frac)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    print("Auto domain bounds (3D):")
    print(f"x: [{xmin:.6f}, {xmax:.6f}]")
    print(f"y: [{ymin:.6f}, {ymax:.6f}]")
    print(f"z: [{zmin:.6f}, {zmax:.6f}]")

    x_grid = np.linspace(xmin, xmax, args.nx, dtype=np.float64)
    y_grid = np.linspace(ymin, ymax, args.ny, dtype=np.float64)
    z_grid = np.linspace(zmin, zmax, args.nz, dtype=np.float64)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")

    metadata = {
        "config": {
            "data_dir": str(data_dir),
            "out_dir": str(out_dir),
            "nx": int(args.nx),
            "ny": int(args.ny),
            "nz": int(args.nz),
            "margin_frac": float(args.margin_frac),
            "kernel_trunc_sigma": float(args.kernel_trunc_sigma),
            "sigma_floor": float(args.sigma_floor),
            "domain": {
                "xmin": float(xmin),
                "xmax": float(xmax),
                "ymin": float(ymin),
                "ymax": float(ymax),
                "zmin": float(zmin),
                "zmax": float(zmax),
            },
        },
        "frames": [],
    }

    print(f"Processing {len(files)} frames -> {out_dir}")
    save_fn = np.savez_compressed if args.save_compressed else np.savez
    save_mode = "compressed" if args.save_compressed else "uncompressed"
    print(f"Save mode: {save_mode}")

    for file_path in tqdm(files):
        data = np.load(file_path)

        pos = np.asarray(data["pos"][:, :3], dtype=np.float64)
        gamma = np.asarray(data["gamma_mag"], dtype=np.float64)
        sigma = np.asarray(data["sigma"], dtype=np.float64)
        vol = np.asarray(data["vol"], dtype=np.float64)

        nodes = np.asarray(data["nodes"][:, :3], dtype=np.float64)
        U_nodes = np.asarray(data["U_nodes"][:, :3], dtype=np.float64)

        pproj = project_particles_to_grid_3d(
            pos=pos,
            gamma=gamma,
            sigma=sigma,
            vol=vol,
            x_grid=x_grid,
            y_grid=y_grid,
            z_grid=z_grid,
            X=X,
            Y=Y,
            Z=Z,
            bounds=bounds,
            kernel_trunc_sigma=args.kernel_trunc_sigma,
            sigma_floor=args.sigma_floor,
        )

        U_grid, used_nodes, dropped_nodes_oob = velocity_to_grid_3d(
            nodes=nodes,
            U_nodes=U_nodes,
            x_grid=x_grid,
            y_grid=y_grid,
            z_grid=z_grid,
            bounds=bounds,
        )

        out_name = out_dir / f"{file_path.stem}_grid.npz"
        save_fn(
            out_name,
            input_grid=pproj["input_grid"],
            input_grid_no_omega=pproj["input_grid_no_omega"],
            input_grid_omega=pproj["input_grid_omega"],
            input_grid_omega4=pproj["input_grid_omega4"],
            U_grid=U_grid,
            x_grid=x_grid.astype(np.float32),
            y_grid=y_grid.astype(np.float32),
            z_grid=z_grid.astype(np.float32),
            domain=np.array([xmin, xmax, ymin, ymax, zmin, zmax], dtype=np.float32),
        )

        frame_id = int(file_path.stem.split("_")[1])
        metadata["frames"].append(
            {
                "frame": frame_id,
                "file": str(out_name),
                "n_particles_total": int(pos.shape[0]),
                "n_particles_used": int(pproj["n_particles_used"]),
                "n_particles_dropped_oob": int(pproj["n_particles_dropped_oob"]),
                "n_particles_dropped_bad_sigma": int(pproj["n_particles_dropped_bad_sigma"]),
                "n_nodes_total": int(nodes.shape[0]),
                "n_nodes_used": int(used_nodes),
                "n_nodes_dropped_oob": int(dropped_nodes_oob),
            }
        )

    meta_path = out_dir / "metadata_autodomain_3d_grid.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done. Wrote {len(metadata['frames'])} 3D grid frames.")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()

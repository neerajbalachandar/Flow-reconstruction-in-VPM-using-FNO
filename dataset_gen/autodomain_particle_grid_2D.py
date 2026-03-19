import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert particle/node frame data to 2D grid with auto domain."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/train/pair_1"),
        help="Folder containing frame_*.npz inputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/train/pair_1_grid_2D_auto"),
        help="Folder for output frame_*_grid.npz files.",
    )
    parser.add_argument("--nx", type=int, default=32, help="Grid resolution in x.")
    parser.add_argument("--ny", type=int, default=32, help="Grid resolution in y.")
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
        help="Gaussian support radius in units of sigma for particle deposition.",
    )
    return parser.parse_args()


def compute_auto_domain(files, margin_frac):
    xmin, ymin = np.inf, np.inf
    xmax, ymax = -np.inf, -np.inf

    for file_path in files:
        data = np.load(file_path)

        pos_xy = data["pos"][:, :2]
        nodes_xy = data["nodes"][:, :2]

        all_xy = np.concatenate([pos_xy, nodes_xy], axis=0)

        xmin = min(xmin, float(all_xy[:, 0].min()))
        xmax = max(xmax, float(all_xy[:, 0].max()))
        ymin = min(ymin, float(all_xy[:, 1].min()))
        ymax = max(ymax, float(all_xy[:, 1].max()))

    x_span = max(xmax - xmin, 1e-12)
    y_span = max(ymax - ymin, 1e-12)

    x_pad = margin_frac * x_span
    y_pad = margin_frac * y_span

    return xmin - x_pad, xmax + x_pad, ymin - y_pad, ymax + y_pad


def gaussian_kernel(dx, dy, h):
    return np.exp(-(dx * dx + dy * dy) / (2.0 * h * h))


def in_domain_mask(points_xy, xmin, xmax, ymin, ymax):
    return (
        (points_xy[:, 0] >= xmin)
        & (points_xy[:, 0] <= xmax)
        & (points_xy[:, 1] >= ymin)
        & (points_xy[:, 1] <= ymax)
    )


def nearest_uniform_index(values, vmin, vmax, n):
    if n <= 1 or vmax <= vmin:
        return np.zeros_like(values, dtype=np.int64)
    scaled = (values - vmin) / (vmax - vmin)
    idx = np.rint(scaled * (n - 1)).astype(np.int64)
    return np.clip(idx, 0, n - 1)


def particles_to_grid(
    pos,
    omega,
    sigma,
    vol,
    x_grid,
    y_grid,
    X,
    Y,
    xmin,
    xmax,
    ymin,
    ymax,
    kernel_trunc_sigma,
):
    nx, ny = X.shape
    grid = np.zeros((3, nx, ny), dtype=np.float32)

    domain_mask = in_domain_mask(pos, xmin, xmax, ymin, ymax)
    pos = pos[domain_mask]
    omega = omega[domain_mask]
    sigma = sigma[domain_mask]
    vol = vol[domain_mask]

    dropped_oob = int((~domain_mask).sum())
    dropped_bad_sigma = 0

    for p in range(pos.shape[0]):
        h = float(sigma[p])
        if not np.isfinite(h) or h <= 0.0:
            dropped_bad_sigma += 1
            continue

        radius = kernel_trunc_sigma * h
        px, py = float(pos[p, 0]), float(pos[p, 1])

        ix0 = np.searchsorted(x_grid, px - radius, side="left")
        ix1 = np.searchsorted(x_grid, px + radius, side="right")
        iy0 = np.searchsorted(y_grid, py - radius, side="left")
        iy1 = np.searchsorted(y_grid, py + radius, side="right")

        if ix0 >= ix1 or iy0 >= iy1:
            continue

        dx = X[ix0:ix1, iy0:iy1] - px
        dy = Y[ix0:ix1, iy0:iy1] - py
        w = gaussian_kernel(dx, dy, h)

        grid[0, ix0:ix1, iy0:iy1] += omega[p] * w
        grid[1, ix0:ix1, iy0:iy1] += sigma[p] * w
        grid[2, ix0:ix1, iy0:iy1] += vol[p] * w

    return grid, int(pos.shape[0]), dropped_oob, dropped_bad_sigma


def velocity_to_grid(nodes, U_nodes, xmin, xmax, ymin, ymax, nx, ny):
    u_grid = np.zeros((2, nx, ny), dtype=np.float32)
    count = np.zeros((nx, ny), dtype=np.float32)

    mask = in_domain_mask(nodes, xmin, xmax, ymin, ymax)
    nodes = nodes[mask]
    U_nodes = U_nodes[mask]
    dropped_oob = int((~mask).sum())

    ix = nearest_uniform_index(nodes[:, 0], xmin, xmax, nx)
    iy = nearest_uniform_index(nodes[:, 1], ymin, ymax, ny)

    np.add.at(u_grid[0], (ix, iy), U_nodes[:, 0])
    np.add.at(u_grid[1], (ix, iy), U_nodes[:, 1])
    np.add.at(count, (ix, iy), 1.0)

    count[count == 0] = 1.0
    u_grid /= count

    return u_grid, int(nodes.shape[0]), dropped_oob


def main():
    args = parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.data_dir.glob("frame_*.npz"))

    if not files:
        raise RuntimeError(f"No frame_*.npz files found in {args.data_dir}")

    xmin, xmax, ymin, ymax = compute_auto_domain(files, args.margin_frac)

    print("Auto domain bounds:")
    print(f"x: [{xmin:.6f}, {xmax:.6f}]")
    print(f"y: [{ymin:.6f}, {ymax:.6f}]")

    x_grid = np.linspace(xmin, xmax, args.nx, dtype=np.float64)
    y_grid = np.linspace(ymin, ymax, args.ny, dtype=np.float64)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

    metadata = {
        "config": {
            "data_dir": str(args.data_dir),
            "out_dir": str(args.out_dir),
            "nx": int(args.nx),
            "ny": int(args.ny),
            "margin_frac": float(args.margin_frac),
            "kernel_trunc_sigma": float(args.kernel_trunc_sigma),
            "domain": {
                "xmin": float(xmin),
                "xmax": float(xmax),
                "ymin": float(ymin),
                "ymax": float(ymax),
            },
        },
        "frames": [],
    }

    print(f"Processing {len(files)} frames -> {args.out_dir}")

    for file_path in tqdm(files):
        data = np.load(file_path)

        pos = data["pos"][:, :2]
        omega = data["gamma_mag"]
        sigma = data["sigma"]
        vol = data["vol"]
        nodes = data["nodes"][:, :2]
        U_nodes = data["U_nodes"][:, :2]

        input_grid, used_particles, dropped_particles_oob, dropped_bad_sigma = (
            particles_to_grid(
                pos=pos,
                omega=omega,
                sigma=sigma,
                vol=vol,
                x_grid=x_grid,
                y_grid=y_grid,
                X=X,
                Y=Y,
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                kernel_trunc_sigma=args.kernel_trunc_sigma,
            )
        )

        U_grid, used_nodes, dropped_nodes_oob = velocity_to_grid(
            nodes=nodes,
            U_nodes=U_nodes,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            nx=args.nx,
            ny=args.ny,
        )

        out_name = args.out_dir / f"{file_path.stem}_grid.npz"
        np.savez(
            out_name,
            input_grid=input_grid,
            U_grid=U_grid,
            domain=np.array([xmin, xmax, ymin, ymax], dtype=np.float32),
        )

        frame_id = int(file_path.stem.split("_")[1])
        metadata["frames"].append(
            {
                "frame": frame_id,
                "file": str(out_name),
                "n_particles_total": int(pos.shape[0]),
                "n_particles_used": used_particles,
                "n_particles_dropped_oob": dropped_particles_oob,
                "n_particles_dropped_bad_sigma": dropped_bad_sigma,
                "n_nodes_total": int(nodes.shape[0]),
                "n_nodes_used": used_nodes,
                "n_nodes_dropped_oob": dropped_nodes_oob,
            }
        )

    meta_path = args.out_dir / "metadata_autodomain.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done. Wrote {len(metadata['frames'])} grid frames.")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()

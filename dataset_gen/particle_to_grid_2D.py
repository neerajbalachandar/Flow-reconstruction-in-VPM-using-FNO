import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Paths / config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Project particle data to 2D grids with precomputed omega channels."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/train/pair_2"),
        help="Folder containing frame_*.npz particle files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/train/pair_2_grid_2D"),
        help="Output folder for frame_*_grid.npz files",
    )
    parser.add_argument("--nx", type=int, default=32, help="Grid resolution in x")
    parser.add_argument("--ny", type=int, default=32, help="Grid resolution in y")
    parser.add_argument("--xmin", type=float, default=0.5, help="Domain min x")
    parser.add_argument("--xmax", type=float, default=7.2, help="Domain max x")
    parser.add_argument("--ymin", type=float, default=-1.25, help="Domain min y")
    parser.add_argument("--ymax", type=float, default=1.25, help="Domain max y")
    parser.add_argument(
        "--kernel-trunc-sigma",
        type=float,
        default=3.0,
        help="Truncation radius in sigma units",
    )
    parser.add_argument(
        "--sigma-floor",
        type=float,
        default=1e-6,
        help="Lower bound for sigma to avoid divide-by-zero",
    )
    return parser.parse_args()


args = parse_args()
data_dir = args.data_dir
out_dir = args.out_dir
out_dir.mkdir(parents=True, exist_ok=True)

xmin, ymin = np.inf, np.inf
xmax, ymax = -np.inf, -np.inf

for file_path in data_dir.glob("frame_*.npz"):
    data = np.load(file_path)
    pos = data["pos"][:, :2]
    nodes = data["nodes"][:, :2] if "nodes" in data else pos
    all_xy = np.concatenate([pos, nodes], axis=0)

    xmin = min(xmin, all_xy[:, 0].min())
    xmax = max(xmax, all_xy[:, 0].max())
    ymin = min(ymin, all_xy[:, 1].min())
    ymax = max(ymax, all_xy[:, 1].max())

print("Global bounds from particles + nodes:")
print(f"x: [{xmin:.5f}, {xmax:.5f}]")
print(f"y: [{ymin:.5f}, {ymax:.5f}]")

# Domain
xmin, xmax = args.xmin, args.xmax
ymin, ymax = args.ymin, args.ymax

Nx, Ny = args.nx, args.ny
KERNEL_TRUNC_SIGMA = args.kernel_trunc_sigma
SIGMA_FLOOR = args.sigma_floor

# Grid
x_grid = np.linspace(xmin, xmax, Nx)
y_grid = np.linspace(ymin, ymax, Ny)
X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")


def zeta(q):
    """
    Base filter kernel zeta(q) used in:
    omega(x) = sum_p Gamma_p * zeta_sigma(x - x_p),
    zeta_sigma = zeta(||x-x_p|| / sigma_p) / sigma_p^2
    """
    return np.exp(-(q ** 2)) / np.pi


def project_particle_channels(pos, gamma, sigma, vol, X, Y, x_grid, y_grid):
    """
    Build both legacy and omega-based channels from particles.

    Returns:
        legacy_input_grid:      (3, Nx, Ny) = [gamma_gauss, sigma_gauss, vol_gauss]
        input_grid_no_omega:    (3, Nx, Ny) = [gamma_gauss, sigma_gauss, pos_density]
        input_grid_omega:       (1, Nx, Ny) = [omega]
        input_grid_omega4:      (4, Nx, Ny) = [omega, gamma_gauss, sigma_gauss, pos_density]
    """
    nx, ny = X.shape

    omega_grid = np.zeros((nx, ny), dtype=np.float32)
    gamma_grid = np.zeros((nx, ny), dtype=np.float32)
    sigma_grid = np.zeros((nx, ny), dtype=np.float32)
    pos_density = np.zeros((nx, ny), dtype=np.float32)
    vol_grid = np.zeros((nx, ny), dtype=np.float32)

    in_domain = (
        (pos[:, 0] >= x_grid[0])
        & (pos[:, 0] <= x_grid[-1])
        & (pos[:, 1] >= y_grid[0])
        & (pos[:, 1] <= y_grid[-1])
    )
    pos = pos[in_domain]
    gamma = gamma[in_domain]
    sigma = sigma[in_domain]
    vol = vol[in_domain]

    for p in range(pos.shape[0]):
        h = max(float(sigma[p]), SIGMA_FLOOR)
        px, py = float(pos[p, 0]), float(pos[p, 1])
        radius = KERNEL_TRUNC_SIGMA * h

        ix0 = np.searchsorted(x_grid, px - radius, side="left")
        ix1 = np.searchsorted(x_grid, px + radius, side="right")
        iy0 = np.searchsorted(y_grid, py - radius, side="left")
        iy1 = np.searchsorted(y_grid, py + radius, side="right")
        if ix0 >= ix1 or iy0 >= iy1:
            continue

        dx = X[ix0:ix1, iy0:iy1] - px
        dy = Y[ix0:ix1, iy0:iy1] - py
        r = np.sqrt(dx * dx + dy * dy)
        q = r / h

        # Physics-based vorticity channel
        zeta_sigma = zeta(q) / (h ** 2)
        omega_grid[ix0:ix1, iy0:iy1] += gamma[p] * zeta_sigma

        # Auxiliary projected channels
        w = np.exp(-(q ** 2))
        gamma_grid[ix0:ix1, iy0:iy1] += gamma[p] * w
        sigma_grid[ix0:ix1, iy0:iy1] += sigma[p] * w
        pos_density[ix0:ix1, iy0:iy1] += w
        vol_grid[ix0:ix1, iy0:iy1] += vol[p] * w

    legacy_input_grid = np.stack([gamma_grid, sigma_grid, vol_grid], axis=0).astype(
        np.float32
    )
    input_grid_no_omega = np.stack([gamma_grid, sigma_grid, pos_density], axis=0).astype(
        np.float32
    )
    input_grid_omega = omega_grid[None, :, :].astype(np.float32)
    input_grid_omega4 = np.stack(
        [omega_grid, gamma_grid, sigma_grid, pos_density], axis=0
    ).astype(np.float32)

    return legacy_input_grid, input_grid_no_omega, input_grid_omega, input_grid_omega4


def nearest_uniform_index(values, vmin, vmax, n):
    if n <= 1 or vmax <= vmin:
        return np.zeros_like(values, dtype=np.int64)
    scaled = (values - vmin) / (vmax - vmin)
    idx = np.rint(scaled * (n - 1)).astype(np.int64)
    return np.clip(idx, 0, n - 1)


def velocity_to_grid(nodes, U_nodes, x_grid, y_grid):
    """
    nodes   : (Nn, 2)
    U_nodes : (Nn, 2)
    """
    nx, ny = len(x_grid), len(y_grid)
    u_grid = np.zeros((2, nx, ny), dtype=np.float32)
    count = np.zeros((nx, ny), dtype=np.float32)

    in_domain = (
        (nodes[:, 0] >= x_grid[0])
        & (nodes[:, 0] <= x_grid[-1])
        & (nodes[:, 1] >= y_grid[0])
        & (nodes[:, 1] <= y_grid[-1])
    )
    nodes = nodes[in_domain]
    U_nodes = U_nodes[in_domain]

    ix = nearest_uniform_index(nodes[:, 0], x_grid[0], x_grid[-1], nx)
    iy = nearest_uniform_index(nodes[:, 1], y_grid[0], y_grid[-1], ny)

    np.add.at(u_grid[0], (ix, iy), U_nodes[:, 0])
    np.add.at(u_grid[1], (ix, iy), U_nodes[:, 1])
    np.add.at(count, (ix, iy), 1.0)

    count[count == 0] = 1.0
    u_grid /= count
    return u_grid


files = sorted(data_dir.glob("frame_*.npz"))
print(f"Processing {len(files)} frames -> grid data")

for file_path in tqdm(files):
    data = np.load(file_path)

    pos = data["pos"][:, :2]
    gamma = data["gamma_mag"]
    sigma = data["sigma"]
    vol = data["vol"]

    legacy_input_grid, input_grid_no_omega, input_grid_omega, input_grid_omega4 = (
        project_particle_channels(
            pos=pos,
            gamma=gamma,
            sigma=sigma,
            vol=vol,
            X=X,
            Y=Y,
            x_grid=x_grid,
            y_grid=y_grid,
        )
    )

    nodes = data["nodes"][:, :2]
    U_nodes = data["U_nodes"][:, :2]
    U_grid = velocity_to_grid(nodes, U_nodes, x_grid, y_grid)

    out_name = out_dir / f"{file_path.stem}_grid.npz"
    np.savez(
        out_name,
        input_grid=legacy_input_grid,
        input_grid_no_omega=input_grid_no_omega,
        input_grid_omega=input_grid_omega,
        input_grid_omega4=input_grid_omega4,
        U_grid=U_grid,
        x_grid=x_grid.astype(np.float32),
        y_grid=y_grid.astype(np.float32),
        domain=np.array([xmin, xmax, ymin, ymax], dtype=np.float32),
    )

print("Done. Grid dataset saved to:", out_dir)

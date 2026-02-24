import numpy as np
from pathlib import Path
from tqdm import tqdm

# Path

data_dir = Path("data/train/pair_1")          # input particle data
out_dir  = Path("data/train/pair_1_grid_2D")     # output grid data
out_dir.mkdir(parents=True, exist_ok=True)

xmin, ymin = np.inf, np.inf
xmax, ymax = -np.inf, -np.inf

for f in data_dir.glob("frame_*.npz"):
    data = np.load(f)
    pos = data["pos"]  # (Np, 2)

    xmin = min(xmin, pos[:, 0].min())
    xmax = max(xmax, pos[:, 0].max())

    ymin = min(ymin, pos[:, 1].min())
    ymax = max(ymax, pos[:, 1].max())

print("Global particle bounds:")
print(f"x: [{xmin:.5f}, {xmax:.5f}]")
print(f"y: [{ymin:.5f}, {ymax:.5f}]")

# Domain
xmin, xmax = 0.5, 2.0
ymin, ymax = -2.1, 0.1

Nx, Ny = 64, 64   # grid resolution

# Grid 

x_grid = np.linspace(xmin, xmax, Nx)
y_grid = np.linspace(ymin, ymax, Ny)
X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

# Kernel

def gaussian_kernel(dx, dy, h):
    return np.exp(-(dx**2 + dy**2) / (2.0 * h**2))


# Particle to Grid

def particles_to_grid(pos, omega, sigma, vol, X, Y):
    """
    pos   : (Np, 2)
    omega : (Np,)
    sigma : (Np,)
    vol   : (Np,)
    """

    Nx, Ny = X.shape
    grid = np.zeros((3, Nx, Ny), dtype=np.float32)

    for p in range(pos.shape[0]):
        dx = X - pos[p, 0]
        dy = Y - pos[p, 1]

        h = sigma[p]
        w = gaussian_kernel(dx, dy, h)

        grid[0] += omega[p] * w
        grid[1] += sigma[p] * w
        grid[2] += vol[p]   * w

    return grid


# Velocity to grid

def velocity_to_grid(nodes, U_nodes, X, Y):
    """
    nodes   : (Nn, 2)
    U_nodes : (Nn, 2)
    """

    u_grid = np.zeros((2, X.shape[0], X.shape[1]), dtype=np.float32)
    count  = np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)

    for i in range(nodes.shape[0]):
        ix = np.argmin(np.abs(x_grid - nodes[i, 0]))
        iy = np.argmin(np.abs(y_grid - nodes[i, 1]))

        u_grid[0, ix, iy] += U_nodes[i, 0]
        u_grid[1, ix, iy] += U_nodes[i, 1]
        count[ix, iy] += 1.0

    count[count == 0] = 1.0
    u_grid /= count

    return u_grid


# main

files = sorted(data_dir.glob("frame_*.npz"))

print(f"Processing {len(files)} frames → grid data")

for f in tqdm(files):
    data = np.load(f)

    # ---------------- INPUT ----------------
    pos   = data["pos"][:, :2]
    omega = data["gamma_mag"]
    sigma = data["sigma"]
    vol   = data["vol"]

    input_grid = particles_to_grid(pos, omega, sigma, vol, X, Y)

    # ---------------- OUTPUT ----------------
    nodes   = data["nodes"][:, :2]
    U_nodes = data["U_nodes"][:, :2]

    U_grid = velocity_to_grid(nodes, U_nodes, X, Y)

    # ---------------- SAVE ----------------
    out_name = out_dir / f"{f.stem}_grid.npz"
    np.savez(
        out_name,
        input_grid=input_grid,
        U_grid=U_grid
    )

print("Done. Grid dataset saved to:", out_dir)

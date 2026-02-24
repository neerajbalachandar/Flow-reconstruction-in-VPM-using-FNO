import numpy as np
from pathlib import Path
from tqdm import tqdm

# Path

data_dir = Path("data/train/pair_1")          # input particle data
out_dir  = Path("data/train/pair_1_grid_3D")     # output grid data
out_dir.mkdir(parents=True, exist_ok=True)

xmin, ymin, zmin = np.inf, np.inf, np.inf
xmax, ymax, zmax = -np.inf, -np.inf, -np.inf

for f in data_dir.glob("frame_*.npz"):
    data = np.load(f)
    pos = data["pos"]  # (Np, 3)

    xmin = min(xmin, pos[:, 0].min())
    xmax = max(xmax, pos[:, 0].max())

    ymin = min(ymin, pos[:, 1].min())
    ymax = max(ymax, pos[:, 1].max())

    zmin = min(zmin, pos[:, 2].min())
    zmax = max(zmax, pos[:, 2].max())

print("Global particle bounds:")
print(f"x: [{xmin:.5f}, {xmax:.5f}]")
print(f"y: [{ymin:.5f}, {ymax:.5f}]")
print(f"z: [{zmin:.5f}, {zmax:.5f}]")


# Physical domain (3D)
xmin, xmax = 0.8, 1.8
ymin, ymax = -2.1, 0.1
zmin, zmax = -0.9, -0.4

Nx, Ny, Nz = 32, 32, 16   # grid resolution (start small in z)

# Grid

x_grid = np.linspace(xmin, xmax, Nx)
y_grid = np.linspace(ymin, ymax, Ny)
z_grid = np.linspace(zmin, zmax, Nz)

X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")

# Kernel

def gaussian_kernel_3d(dx, dy, dz, h):
    return np.exp(-(dx**2 + dy**2 + dz**2) / (2.0 * h**2))


# Particle to grid

def particles_to_grid_3d(pos, omega, sigma, vol, X, Y, Z):
    """
    pos   : (Np, 3)
    omega : (Np,)
    sigma : (Np,)
    vol   : (Np,)
    """

    Nx, Ny, Nz = X.shape
    grid = np.zeros((3, Nx, Ny, Nz), dtype=np.float32)

    for p in range(pos.shape[0]):
        dx = X - pos[p, 0]
        dy = Y - pos[p, 1]
        dz = Z - pos[p, 2]

        h = sigma[p]
        w = gaussian_kernel_3d(dx, dy, dz, h)

        grid[0] += omega[p] * w
        grid[1] += sigma[p] * w
        grid[2] += vol[p]   * w

    return grid

# velocity to grid

def velocity_to_grid_3d(nodes, U_nodes, x_grid, y_grid, z_grid):
    """
    nodes   : (Nn, 3)
    U_nodes : (Nn, 3)
    """

    u_grid = np.zeros(
        (3, len(x_grid), len(y_grid), len(z_grid)),
        dtype=np.float32
    )
    count = np.zeros(
        (len(x_grid), len(y_grid), len(z_grid)),
        dtype=np.float32
    )

    for i in range(nodes.shape[0]):
        ix = np.argmin(np.abs(x_grid - nodes[i, 0]))
        iy = np.argmin(np.abs(y_grid - nodes[i, 1]))
        iz = np.argmin(np.abs(z_grid - nodes[i, 2]))

        u_grid[0, ix, iy, iz] += U_nodes[i, 0]
        u_grid[1, ix, iy, iz] += U_nodes[i, 1]
        u_grid[2, ix, iy, iz] += U_nodes[i, 2]
        count[ix, iy, iz] += 1.0

    count[count == 0] = 1.0
    u_grid /= count

    return u_grid


# Main

files = sorted(data_dir.glob("frame_*.npz"))
print(f"Processing {len(files)} frames → 3D grid data")

for f in tqdm(files):
    data = np.load(f)

    # ---------------- INPUT ----------------
    pos   = data["pos"][:, :3]
    omega = data["gamma_mag"]
    sigma = data["sigma"]
    vol   = data["vol"]

    input_grid = particles_to_grid_3d(pos, omega, sigma, vol, X, Y, Z)

    # ---------------- OUTPUT ----------------
    nodes   = data["nodes"][:, :3]
    U_nodes = data["U_nodes"][:, :3]

    U_grid = velocity_to_grid_3d(nodes, U_nodes, x_grid, y_grid, z_grid)

    # ---------------- SAVE ----------------
    out_name = out_dir / f"{f.stem}_grid.npz"
    np.savez(
        out_name,
        input_grid=input_grid,
        U_grid=U_grid
    )

print("Done. 3D grid dataset saved to:", out_dir)

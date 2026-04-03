import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from neuralop.layers.embeddings import GridEmbedding2D


# Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "data", "train", "pair_1_grid_3D")
)

print("Current working directory :", os.getcwd())
print("Resolved data directory   :", DATA_DIR)
print("Directory exists          :", os.path.isdir(DATA_DIR))


# File path 

files = sorted(
    glob.glob(os.path.join(DATA_DIR, "frame_*_grid.npz")),
    key=lambda f: int(os.path.basename(f).split("_")[1])
)

train_files = [
    f for f in files
    if int(os.path.basename(f).split("_")[1]) <= 300
]

print("Total grid files :", len(files))
print("Train grid files :", len(train_files))

if len(train_files) == 0:
    raise RuntimeError("No training files found.")


# Dataset

class VPMGridDataset3D(torch.utils.data.Dataset):
    """
    a(x,y,z) -> u(x,y,z)

    input_grid : (C, Nx, Ny, Nz)
    U_grid     : (3, Nx, Ny, Nz)
    """

    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        x = torch.tensor(data["input_grid"], dtype=torch.float32)
        y = torch.tensor(data["U_grid"], dtype=torch.float32)

        return {"x": x, "y": y}


#Loader 

train_dataset = VPMGridDataset3D(train_files)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,     # safest for 3D grids
    shuffle=True
)

batch = next(iter(train_loader))
x, y = batch["x"], batch["y"]

print("Input x shape :", x.shape)   # (B, C, Nx, Ny, Nz)
print("Output y shape:", y.shape)   # (B, 3, Nx, Ny, Nz)


# 2D positional embedding (2D-Slice)

sample_id = 0
z_slice = x.shape[-1] // 2     # mid-plane slice

# Extract a 2D slice: (B, C, Nx, Ny)
x_slice = x[:, :, :, :, z_slice]

print("2D slice shape:", x_slice.shape)

pos_embedding = GridEmbedding2D(in_channels=x_slice.shape[1])
x_pe = pos_embedding(x_slice)

print("With positional embedding:", x_pe.shape)
# (B, C+2, Nx, Ny)




plt.figure(figsize=(12, 4))

# ---- Input: vorticity (ω) slice ----
plt.subplot(1, 4, 1)
plt.imshow(x_slice[sample_id, 0].cpu())
plt.title("Input ω(x,y) at z₀")
plt.colorbar()

# ---- Output: velocity magnitude slice ----
speed = torch.sqrt(
    y[sample_id, 0, :, :, z_slice]**2 +
    y[sample_id, 1, :, :, z_slice]**2 +
    y[sample_id, 2, :, :, z_slice]**2
)

plt.subplot(1, 4, 2)
plt.imshow(speed.cpu())
plt.title("|u(x,y)| at z₀")
plt.colorbar()

# ---- Positional embedding: x-coordinate ----
plt.subplot(1, 4, 3)
plt.imshow(x_pe[sample_id, -2].cpu())
plt.title("x-coordinate embedding")

# ---- Positional embedding: y-coordinate ----
plt.subplot(1, 4, 4)
plt.imshow(x_pe[sample_id, -1].cpu())
plt.title("y-coordinate embedding")

plt.tight_layout()
plt.show()

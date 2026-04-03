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
    os.path.join(BASE_DIR, "..", "data", "train", "pair_1_grid_2D")
)

print("Current working directory :", os.getcwd())
print("Resolved data directory   :", DATA_DIR)
print("Directory exists          :", os.path.isdir(DATA_DIR))


files = sorted(
    glob.glob(os.path.join(DATA_DIR, "frame_*_grid.npz")),
    key=lambda f: int(os.path.basename(f).split("_")[1])
)

train_files = [f for f in files if int(os.path.basename(f).split("_")[1]) <= 300]

print("Total grid files :", len(files))
print("Train grid files :", len(train_files))

if len(train_files) == 0:
    raise RuntimeError(
        f"No training files found in {DATA_DIR}\n"
        "Expected files like: frame_XXX_grid.npz"
    )


# Dataset

class VPMGridDataset(Dataset):
    """
    a(x,y) -> u(x,y)

    input_grid : (C, Nx, Ny)
    U_grid     : (2, Nx, Ny)
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


# Loader 

train_dataset = VPMGridDataset(train_files)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    drop_last=True
)

batch = next(iter(train_loader))
x, y = batch["x"], batch["y"]

print("Input x shape :", x.shape)   # (B, C, Nx, Ny)
print("Output y shape:", y.shape)   # (B, 2, Nx, Ny)


# Positional embedding

pos_embedding = GridEmbedding2D(in_channels=x.shape[1])
x_pe = pos_embedding(x)

print("With positional embedding:", x_pe.shape)




sample_id = 0

plt.figure(figsize=(10, 4))

# Input: vorticity (channel 0)
plt.subplot(1, 3, 1)
plt.imshow(x[sample_id, 0].cpu())
plt.title("Input ω(x,y)")
plt.colorbar()

# Output: velocity magnitude
speed = torch.sqrt(y[sample_id, 0]**2 + y[sample_id, 1]**2)
plt.subplot(1, 3, 2)
plt.imshow(speed.cpu())
plt.title("|u(x,y)|")
plt.colorbar()

# Positional embedding (x-coordinate)
plt.subplot(1, 3, 3)
plt.imshow(x_pe[sample_id, -2].cpu())
plt.title("x-coordinate embedding")

plt.tight_layout()
plt.show()

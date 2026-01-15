import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from neuralop.layers.embeddings import GridEmbedding2D
import glob
import os


data_dir = "data/train/pair_1_grid"   # preprocessed data

files = sorted(
    glob.glob(os.path.join(data_dir, "frame_*_grid.npz")),
    key=lambda f: int(os.path.basename(f).split("_")[1])
)

train_files = [f for f in files if int(f.split("_")[1]) <= 300]

print("Total grid files:", len(files))
print("Train grid files:", len(train_files))

assert len(train_files) > 0


data_dir = "data/train/pair_1_grid"

files = sorted(
    glob.glob(os.path.join(data_dir, "frame_*_grid.npz")),
    key=lambda f: int(os.path.basename(f).split("_")[1])
)

train_files = [f for f in files if int(f.split("_")[1]) <= 300]

print("Total grid files:", len(files))
print("Train grid files:", len(train_files))

assert len(train_files) > 0

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

        return {
            "x": x,   # (C, Nx, Ny)
            "y": y    # (2, Nx, Ny)
        }


train_dataset = VPMGridDataset(train_files)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True
)

batch = next(iter(train_loader))
x, y = batch["x"], batch["y"]

print("Input x shape :", x.shape)   # (B, C, Nx, Ny) - C input channels
print("Output y shape:", y.shape)   # (B, 2, Nx, Ny)



pos_embedding = GridEmbedding2D(in_channels=x.shape[1])
x_pe = pos_embedding(x)

print("With positional embedding:", x_pe.shape)


sample_id = 0

plt.figure(figsize=(10, 4))

# Input: vorticity (channel 0)
plt.subplot(1, 3, 1)
plt.imshow(x[sample_id, 0])
plt.title("Input ω(x,y)")
plt.colorbar()

# Output: velocity magnitude
speed = torch.sqrt(y[sample_id, 0]**2 + y[sample_id, 1]**2)
plt.subplot(1, 3, 2)
plt.imshow(speed)
plt.title("|u(x,y)|")
plt.colorbar()

# Positional embedding (x-coordinate)
plt.subplot(1, 3, 3)
plt.imshow(x_pe[sample_id, -2])
plt.title("x-coordinate embedding")

plt.tight_layout()
plt.show()


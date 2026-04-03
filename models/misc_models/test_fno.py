import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.losses import LpLoss
from neuralop.utils import count_model_params

# ======================================================
# DEVICE
# ======================================================
device = "cpu"

# ======================================================
# PATHS
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_ROOT = os.path.join(BASE_DIR, "..", "data", "train")

SIM_DIRS = [
    os.path.join(TRAIN_ROOT, "pair_1_grid_2D"),
    os.path.join(TRAIN_ROOT, "pair_2_grid_2D"),
]

for d in SIM_DIRS:
    assert os.path.isdir(d), f"Missing directory: {d}"


def split_simulation_frames(sim_dir, train_frac=0.8):
    files = sorted(
        glob.glob(os.path.join(sim_dir, "frame_*_grid.npz")),
        key=lambda f: int(os.path.basename(f).split("_")[1])
    )

    n_total = len(files)
    n_train = int(train_frac * n_total)

    train_files = files[:n_train]
    test_files  = files[n_train:]

    return train_files, test_files


train_files = []
test_files  = []

for sim_dir in SIM_DIRS:
    tr, te = split_simulation_frames(sim_dir, train_frac=0.8)
    train_files.extend(tr)
    test_files.extend(te)

print("===================================")
print(f"Simulations used   : {len(SIM_DIRS)}")
print(f"Train samples     : {len(train_files)}")
print(f"Test samples      : {len(test_files)}")
print("===================================")

assert len(train_files) > 0
assert len(test_files) > 0


# ======================================================
# DATASET
# ======================================================
class VPMGridDataset(Dataset):
    """
    Operator dataset:
        a(x,y) -> u(x,y)

    input_grid : (3, Nx, Ny)   [ω, σ, vol]
    U_grid     : (2, Nx, Ny)   [u_x, u_y]
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


train_dataset = VPMGridDataset(train_files)
test_dataset  = VPMGridDataset(test_files)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    drop_last=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)

test_loaders = {32: test_loader}

# ======================================================
# MODEL (FNO)
# ======================================================
model = FNO(
    n_modes=(20, 20),
    in_channels=3,
    out_channels=2,
    hidden_channels=64,
    projection_channel_ratio=4,
).to(device)

print(model)
print("Total parameters:", count_model_params(model))

# ======================================================
# OPTIMIZER & LOSS
# ======================================================
optimizer = AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=40
)

l2loss = LpLoss(d=2, p=2)

# ======================================================
# TRAIN
# ======================================================
trainer = Trainer(
    model=model,
    n_epochs=40,
    device=device,
    data_processor=None,
    wandb_log=False,
    eval_interval=5,
    verbose=True,
)

trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    training_loss=l2loss,
    eval_losses={"l2": l2loss},
)

# ======================================================
# QUALITATIVE TEST VISUALIZATION
# ======================================================
model.eval()

sample = test_dataset[0]
x = sample["x"].unsqueeze(0).to(device)
y_true = sample["y"]

with torch.no_grad():
    y_pred = model(x)[0].cpu()

speed_true = torch.sqrt(y_true[0]**2 + y_true[1]**2)
speed_pred = torch.sqrt(y_pred[0]**2 + y_pred[1]**2)
error = torch.norm(y_pred - y_true, dim=0)

plt.figure(figsize=(14, 4))

plt.subplot(1, 4, 1)
plt.imshow(x[0, 0].cpu())
plt.title("Input ω(x,y)")
plt.colorbar()

plt.subplot(1, 4, 2)
plt.imshow(speed_true)
plt.title("True |u|")
plt.colorbar()

plt.subplot(1, 4, 3)
plt.imshow(speed_pred)
plt.title("Predicted |u|")
plt.colorbar()

plt.subplot(1, 4, 4)
plt.imshow(error)
plt.title("Pointwise error")
plt.colorbar()

plt.tight_layout()
plt.show()

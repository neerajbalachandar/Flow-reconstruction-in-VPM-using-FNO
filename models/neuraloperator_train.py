import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.losses import LpLoss
from neuralop.utils import count_model_params


device = 'cpu'

# ======================================================
# PATHS (same logic as dataset_gen)
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "data", "train", "pair_1_grid_2D")
)

files = sorted(
    glob.glob(os.path.join(DATA_DIR, "frame_*_grid.npz")),
    key=lambda f: int(os.path.basename(f).split("_")[1])
)

train_files = [f for f in files if int(os.path.basename(f).split("_")[1]) <= 300]
test_files  = [f for f in files if int(os.path.basename(f).split("_")[1]) > 300]

print(f"Train samples: {len(train_files)}")
print(f"Test samples : {len(test_files)}")

assert len(train_files) > 0 and len(test_files) > 0


# ======================================================
# DATASET
# ======================================================
class VPMGridDataset(Dataset):
    """
    Operator dataset:
        a(x,y) -> u(x,y)

    input_grid : (3, Nx, Ny)  [ω, σ, vol]
    U_grid     : (2, Nx, Ny)  [u_x, u_y]
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
    batch_size=10,
    shuffle=False
) #choosing a higher test batch size decreases loss significantly but is that a good choice fundamentally? ------------------------------------------

test_loaders = {32: test_loader}  # mimic neuralop API


# ======================================================
# MODEL (FNO)
# ======================================================
model = FNO(
    n_modes=(20,20),#higher        # spectral modes
    in_channels=3,           # ω, σ, vol
    out_channels=2,          # u_x, u_y
    hidden_channels=64, #higher
    projection_channel_ratio=6, #higher significant, varying wihtin epochs
)

model = model.to(device)

print(model)
print("Total parameters:", count_model_params(model))


# ======================================================
# OPTIMIZER & LOSSES
# ======================================================
optimizer = AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=30
)

l2loss = LpLoss(d=2, p=2)

train_loss = l2loss
eval_losses = {"l2": l2loss}


# ======================================================
# TRAINER
# ======================================================
trainer = Trainer(
    model=model,
    n_epochs=40,
    device=device,
    data_processor=None,   # IMPORTANT: you already preprocessed
    wandb_log=False,
    eval_interval=5,
    verbose=True,
)

trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    training_loss=train_loss,
    eval_losses=eval_losses,
)


# ======================================================
# QUALITATIVE EVALUATION
# ======================================================
model.eval()
sample = test_dataset[0]

x = sample["x"].unsqueeze(0).to(device)
y_true = sample["y"]

with torch.no_grad():
    y_pred = model(x)[0].cpu()

speed_true = torch.sqrt(y_true[0]**2 + y_true[1]**2)
speed_pred = torch.sqrt(y_pred[0]**2 + y_pred[1]**2)

plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(x[0, 0].cpu())
plt.title("Input ω(x,y)")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(speed_true)
plt.title("True |u|")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(speed_pred)
plt.title("Predicted |u|")
plt.colorbar()

plt.tight_layout()
plt.show()

import os
import numpy as np

# ====== PATHS ======
folder_path = "train/pair_1_fno_16/"
output_file = "pair1_fno_16.txt"
# ===================

files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npz")])

with open(output_file, "w") as f:

    f.write(f"Found {len(files)} .npz files\n\n")

    for fname in files:
        file_path = os.path.join(folder_path, fname)

        f.write("=" * 80 + "\n")
        f.write(f"FILE: {fname}\n")

        data = np.load(file_path)
        f.write(f"Keys inside: {data.files}\n")

        for key in data.files:
            arr = data[key]

            f.write(f"\n  Key: {key}\n")
            f.write(f"    Shape : {arr.shape}\n")
            f.write(f"    Dtype : {arr.dtype}\n")
            f.write(f"    Min   : {np.min(arr)}\n")
            f.write(f"    Max   : {np.max(arr)}\n")
            f.write(f"    Mean  : {np.mean(arr)}\n")

            preview = arr.flatten()[:10]
            f.write(f"    First values: {preview}\n")

        f.write("\n")

print(f"Saved inspection to: {output_file}")
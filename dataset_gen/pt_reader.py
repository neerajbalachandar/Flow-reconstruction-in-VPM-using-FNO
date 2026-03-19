import torch

data = torch.load("/home/dysco/Neeraj/neuralop_project/data/darcy_train_16.pt", map_location="cpu")

print(type(data))

if isinstance(data, dict):
    print("Keys:", data.keys())
    for k, v in data.items():
        print("\nKey:", k)
        print("Type:", type(v))
        if torch.is_tensor(v):
            print("Shape:", v.shape)
            print("Dtype:", v.dtype)
            print("Sample values:", v.flatten()[:10])
        else:
            print("Value:", v)

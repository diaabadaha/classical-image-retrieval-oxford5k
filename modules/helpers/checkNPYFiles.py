import numpy as np

data = np.load("dataset/descriptors/all_souls_000040.npy")
data = np.load("dataset/codebook/codebook.npy")
data = np.load("dataset/vbow/idf.npy")
data = np.load("dataset/vbow/vbowTfidf.npy")

print(data)
print(f"Shape of the array: {data.shape}")
print(f"Data type of the array: {data.dtype}")

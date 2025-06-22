import numpy as np
import pandas as pd
import h5py
from scipy import ndimage

# Open HDF5 file in read mode
with h5py.File('Data/Asg9/wk9_assignment.h5', 'r') as f:
    # List all groups
    print("Keys:", list(f.keys()))
    
    isOccupied = f['isOccupied'][:]
    thermal_maps = f['thermal_maps'][:]


sigma = np.linspace(0.1, 4.0, 40)
threshold = np.linspace(0.5, 1.0, 501)
max_accuracy = 0
best_sigma = 1
best_threshold = 0.1

for i in sigma:
    max_response = []
    for map in range(len(isOccupied)):
        max_response.append(np.max(np.abs(ndimage.gaussian_laplace(thermal_maps[map], i))))

    for t in threshold:
        pred = (max_response >= t).astype(int)

        result = isOccupied == pred

        accuracy = np.sum(result) / len(isOccupied)

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_sigma = i
            best_threshold = t

print(f"Optimal sigma: {best_sigma}, Optimal threshold: {best_threshold}, Accuracy: {max_accuracy}")


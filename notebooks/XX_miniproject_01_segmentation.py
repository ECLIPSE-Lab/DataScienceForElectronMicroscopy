#%%
# pip install foundry-ml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
file_path = '/mnt/data/datascience_miniprojects/image_datasets/Au_np_with_varying_substrate/images/image_batch_46357498354_20240606/'
file_names = os.listdir(file_path)[0]

import h5py as h5

with h5.File(file_path + file_names, 'r') as f:
    d = f['train_batch'][...]
    m = f['mask_batch'][...]
    
# %%
d.shape
# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i in range(3):
    axes[i].imshow(d[i])
    axes[i].axis('off')
    axes[i].set_title(f'Image {i+1}')
plt.tight_layout()
plt.show()
# Plot first 3 masks
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i in range(3):
    axes[i].imshow(m[i])
    axes[i].axis('off') 
    axes[i].set_title(f'Mask {i+1}')
plt.tight_layout()
plt.show()

# %%

# ==============================================================================
# This script is used to save images of the clusters. 
# The cluster sizes are saved to a csv file.
# ==============================================================================

import os
import h5py
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# open the file
with h5py.File("cluster_labels.hdf5", 'r') as f:
    shards = f['shards'][:]
    labels = f['labels'][:]
    filenames = f['filenames'][:]

path_to_images = [os.path.join("./run_2024-02-08-hdf5-test", path) for path in os.listdir("./run_2024-02-08-hdf5-test")]


num_clusters = 100
cluster_sizes = np.zeros(num_clusters)

for j in range(num_clusters):

    idx = np.where(labels == j)[0]
    cluster_size = len(idx)
    cluster_sizes[j] = cluster_size
    
    if cluster_size > 100:
        idx = np.random.choice(idx, 100, replace=False)
    

    keys_shard = shards[idx]
    keys_shard = [key.decode() for key in keys_shard]
    keys_ = filenames[idx]
    keys_ = [key.decode() for key in keys_]

    images = []
    for shard, key in zip(keys_shard, keys_):
        with h5py.File(os.path.join("./run_2024-02-08-hdf5-test", f"{shard}.hdf5"), 'r') as f:
            images.append(f[key][:])
        



    images = [torchvision.io.decode_jpeg(torch.tensor(image)) for image in images]


    fig, axs = plt.subplots(10, 10, figsize=(20, 20))

    for i, ax in enumerate(axs.flatten()):
        try:
            ax.imshow(images[i].permute(1, 2, 0))
        except IndexError:
            pass
        ax.axis('off')
    
    plt.suptitle(f'Cluster {j} - Size: {cluster_size}', fontsize=50, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'clusters/cluster_{j}.png')
    plt.close()


df = pd.DataFrame(cluster_sizes.astype(int))
df.to_csv('cluster_sizes.csv', index=True, header=False)
import h5py

path_to_features = "/Users/ima029/Desktop/SCAMPI/Repository/data/zip scrapings (huge)/features.hdf5"

with h5py.File(path_to_features, 'r') as f:
    features = f['features'][:]
    filenames = f['filenames'][:]

from sklearn.cluster import MiniBatchKMeans

n_clusters = 100
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(features)

labels = kmeans.predict(features)

path_to_images = "/Users/ima029/Desktop/SCAMPI/Repository/data/zip scrapings (huge)/run_2024-02-08-hdf5-test/shard_0.hdf5"

with h5py.File(path_to_images, 'r') as f:
    keys = list(f.keys())

import numpy as np


import torch
import torchvision
import matplotlib.pyplot as plt


for j in range(100):

    keys_ = filenames[np.where(labels == j)]
    keys_ = [key.decode() for key in keys_]

    with h5py.File(path_to_images, 'r') as f:
        images = [f[key][:] for key in keys_]


    images = [torchvision.io.decode_jpeg(torch.tensor(image)) for image in images]


    fig, axs = plt.subplots(10, 10, figsize=(20, 20))

    for i, ax in enumerate(axs.flatten()):
        try:
            ax.imshow(images[i].permute(1, 2, 0))
        except IndexError:
            pass
        ax.axis('off')
    plt.savefig(f'cluster_{j}.png')

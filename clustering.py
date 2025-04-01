# This script clusters the features extracted from images using KMeans clustering.
# It matches the features with the corresponding images and saves the cluster labels and filenames in an HDF5 file.

import os
import h5py
import numpy as np
from sklearn.cluster import MiniBatchKMeans

path_to_features = [os.path.join("embeddings", path) for path in os.listdir("embeddings")]
path_to_images = [os.path.join("./run_2024-02-08-hdf5-test", path) for path in os.listdir("./run_2024-02-08-hdf5-test")]

# match the features with the images
paths = []
for path in path_to_features:
    shard = os.path.basename(path).split(".")[0]
    for image_path in path_to_images:
        if shard == os.path.basename(image_path).split(".")[0]:
            paths.append((shard, path, image_path))

shards = []
features = []
filenames = []

for path in paths:
    shard, feat_path, file_path = path
    with h5py.File(feat_path, 'r') as f:
        features.append(f['features'][:])
    with h5py.File(file_path, 'r') as f:
        filenames.append(list(f.keys()))
    shards.append([shard] * len(filenames[-1]))

shards = np.concatenate(shards, axis=0)
features = np.concatenate(features, axis=0)
filenames = np.concatenate(filenames, axis=0)

n_clusters = 100
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(features)

labels = kmeans.predict(features)

# save filenames and labels
with h5py.File("cluster_labels.hdf5", 'w') as f:
    f.create_dataset('shards', data=shards.astype('S'))
    f.create_dataset('labels', data=labels)
    f.create_dataset('filenames', data=filenames.astype('S'))

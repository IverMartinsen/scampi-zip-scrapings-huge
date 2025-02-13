import os
import glob
import h5py
import shutil
import numpy as np
import pandas as pd

path = "/Users/ima029/Desktop/zip scrapings (huge)/run_2024-02-08"
files = glob.glob(path + "/**/*.jpg")
print(f"Number of files in total: {len(files)}")

destination = "run_2024-02-08 curated"
os.makedirs(destination, exist_ok=True)


# filter out black, blurry, artifact, and multi-object clusters
cluster_stats = pd.read_csv("./clusters/cluster_stats.csv", sep=";")
cluster_stats['clean'] = (cluster_stats['black'] != True) * (cluster_stats['blur'] != True) * (cluster_stats['artifact'] != True) * (cluster_stats['multi'] != True)
clean_clusters = cluster_stats[cluster_stats['clean'] == True].index.values

print(f"Number of clean images: {cluster_stats['size'][clean_clusters].sum()}")

# open the file
with h5py.File("cluster_labels.hdf5", 'r') as f:
    shards = f['shards'][:]
    labels = f['labels'][:]
    files_to_copy = f['filenames'][:]

files_to_copy = np.array([f.decode() for f in files_to_copy])
files_to_copy = files_to_copy[np.where(np.isin(labels, clean_clusters))[0]]

# delete images in path that are not in filenames
print(f"Number of files to copy: {len(files_to_copy)}")

def copy_file(file):
    if os.path.basename(file) in files_to_copy:
        shutil.copy(file, destination)

for file in files:
    copy_file(file)
    
print(f"Number of files copied: {len(os.listdir(destination))}")

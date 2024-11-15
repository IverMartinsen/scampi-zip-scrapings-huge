import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with h5py.File("cluster_labels.hdf5", "r") as f:
    labels = f["labels"][:]
    filenames = f["filenames"][:]

filenames = [f.decode("utf-8") for f in filenames]
filenames = np.array(filenames)

metadata = pd.read_csv("metadata.csv")

for i in range(100):
    idx = np.where(labels == i)[0]
    files = filenames[idx]
    wells = metadata.loc[metadata["file"].isin(files), "well"]
    wells, counts = np.unique(wells, return_counts=True)
    pd.DataFrame({"well": wells, "count": counts}).to_csv(f"cluster_{i}_well_distribution.csv", index=False)
    
    plt.figure()
    plt.bar(wells, counts)
    plt.title(f"Cluster {i} ({len(wells)} wells)")
    plt.xlabel("Well", fontsize=8)
    plt.xticks(rotation=90, fontsize=2)
    plt.ylabel("Count", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"cluster_{i}_well_distribution.png", dpi=300)
    plt.close()
    
    plt.figure()
    plt.hist(counts, bins=20)
    plt.title(f"Cluster {i} ({len(wells)} wells)")
    plt.xlabel("Count", fontsize=8)
    plt.xticks(fontsize=8)
    plt.ylabel("Frequency", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"cluster_{i}_well_distribution_hist.png", dpi=300)
    plt.close()
    

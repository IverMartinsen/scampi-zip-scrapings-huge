# =============================================================================
# Extract metadata from the file names and save it to a csv file.
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from difflib import SequenceMatcher

path = "./run_2024-02-08"

folders = os.listdir(path)
folders = [folder for folder in folders if os.path.isdir(os.path.join(path, folder))]
folders.sort()

files = []

for folder in folders:
    files += [os.path.basename(file) for file in os.listdir(os.path.join(path, folder))]

metadata = np.zeros((len(files), 4), dtype=np.object)

for i, f in enumerate(files):
    # replace "__" with "_"
    file = f.replace("__", "_")
    well_id = "_".join(file.split("_")[1:-4])
    depth = file.split("_")[-4]
    unknown_metadata = "_".join(tuple(file.split("_")[-3:]))
    
    if well_id == "25-11-5":
        a = well_id.split("-")[0]
        remains = "-".join(well_id.split("-")[1:])
    else:
        a = well_id.split("_")[0]
        remains = "_".join(well_id.split("_")[1:])
    b = remains.split("-")[0]
    remains = "-".join(remains.split("-")[1:])
    remains = "-".join(remains.split("_"))
    well_id = f"{a}/{b}-{remains}"
    
    if well_id == "15/9-21-S-4928-mDC":
        well_id = "15/9-21-S"
        depth = "4928"
    
    depth = depth.replace(",", ".")
    depth = depth.strip("m")
    if depth == "11-13" or depth == "R" or depth == "S":
        depth = "0"
    if depth == "4152-9":
        depth = "4152.9"
    try:
        depth = float(depth)
    except ValueError:
        print(f"Error: {f}, {well_id}, {depth}")
    
    metadata[i] = [f, well_id, depth, unknown_metadata]

df = pd.DataFrame(metadata, columns=["file", "well_id", "depth", "unknown_metadata"])

global_well_stats = pd.read_csv("global_well_stats.csv")
global_wells = np.unique(global_well_stats["well"])
num_scans = global_well_stats["slides"]



local_wells, y = np.unique(df["well_id"], return_counts=True)

# compute string similarity


local_wells_new = []

for i, well in enumerate(local_wells):
    if well not in global_wells:
        print(f"{well} not in global well stats")
        # find the most similar well
        similarities = []
        for well2 in global_wells:
            s = SequenceMatcher(None, well, well2)
            similarities.append(s.ratio())
        closest = global_wells[np.argmax(similarities)]
        print(f"Closest match: {closest}")
        local_wells_new.append(closest)
    else:
        local_wells_new.append(well)

mapping = dict(zip())
df["well"] = df["well_id"].map(mapping)

df.to_csv("metadata.csv", index=False)

df = pd.read_csv("metadata.csv")

wells, counts = np.unique(df["well"], return_counts=True)

mapping = dict(zip(global_wells, num_scans))

scans = np.array([mapping[well] for well in wells])


plt.figure()
plt.scatter(scans, counts)
plt.xlabel("Scans", fontsize=8)
plt.xticks(fontsize=8)
plt.ylabel("Count", fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig("scans_vs_count.png", dpi=300)
plt.close()


plt.figure()
plt.bar(wells[:114], counts[:114])
plt.xticks(rotation=90, fontsize=2)
plt.yticks(fontsize=4)
plt.ylim(0, 35000)
plt.savefig("wells_part1.png", dpi=600)
plt.close()

plt.figure()
plt.bar(wells[114:228], counts[114:228])
plt.xticks(rotation=90, fontsize=2)
plt.yticks(fontsize=4)
plt.ylim(0, 35000)
plt.savefig("wells_part2.png", dpi=600)
plt.close()

plt.figure()
plt.bar(wells[228:], counts[228:])
plt.xticks(rotation=90, fontsize=2)
plt.yticks(fontsize=4)
plt.ylim(0, 35000)
plt.savefig("wells_part3.png", dpi=600)
plt.close()



pd.DataFrame({"well_id": wells, "count": counts, "scans": scans}).to_csv("wells.csv", index=False)

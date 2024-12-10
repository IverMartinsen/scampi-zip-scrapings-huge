# =============================================================================
# Extract metadata from the file names and save it to a csv file.
# =============================================================================

import os
import numpy as np
import pandas as pd
from difflib import SequenceMatcher

path = "./run_2024-02-08"

folders = os.listdir(path)
folders = [folder for folder in folders if os.path.isdir(os.path.join(path, folder))]
folders.sort()

files = []

for folder in folders:
    files += [os.path.basename(file) for file in os.listdir(os.path.join(path, folder))]

metadata = np.zeros((len(files), 4), dtype=np.object)

# traverse the files and extract metadata
# lots of edge cases
for i, f in enumerate(files):
    # some files use double underscore...
    file = f.replace("__", "_") # replace "__" with "_"
    # split the file name into three parts: well, depth, unknown_metadata
    well = "_".join(file.split("_")[1:-4])
    depth = file.split("_")[-4]
    unknown_metadata = "_".join(tuple(file.split("_")[-3:]))
    # reformating the well
    if well == "25-11-5":
        a = well.split("-")[0]
        remains = "-".join(well.split("-")[1:])
    else:
        a = well.split("_")[0]
        remains = "_".join(well.split("_")[1:])
    b = remains.split("-")[0]
    remains = "-".join(remains.split("-")[1:])
    remains = "-".join(remains.split("_"))
    well = f"{a}/{b}-{remains}"
    
    if well == "15/9-21-S-4928-mDC":
        well = "15/9-21-S"
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
        print(f"Error: {f}, {well}, {depth}")
    
    metadata[i] = [f, well, depth, unknown_metadata]

df = pd.DataFrame(metadata, columns=["file", "well", "depth", "unknown_metadata"])


# import the global well stats
global_well_stats = pd.read_csv("global_well_stats.csv")
glo_wells = np.unique(global_well_stats["well"])
loc_wells = np.unique(df["well"])
#num_scans = global_well_stats["slides"]

# match the local wells to the global wells names
loc_wells_new = []

for i, well in enumerate(loc_wells):
    if well not in glo_wells:
        print(f"{well} not in global well stats")
        # find the most similar well
        similarities = []
        for well2 in glo_wells:
            s = SequenceMatcher(None, well, well2)
            similarities.append(s.ratio())
        closest = glo_wells[np.argmax(similarities)]
        print(f"Closest match: {closest}")
        loc_wells_new.append(closest)
    else:
        loc_wells_new.append(well)

mapping = dict(zip(loc_wells, loc_wells_new))
df["well"] = df["well"].map(mapping)

df.to_csv("metadata.csv", index=False)

# compute the number unique depths for each well
wells, counts = np.unique(df["well"], return_counts=True)
scans = np.zeros(len(wells), dtype=np.int)

for i, well in enumerate(wells):
    depths = np.unique(df[df["well"] == well]["depth"])
    scans[i] = len(depths)

pd.DataFrame({"well": wells, "count": counts, "scans": scans}).to_csv("wells.csv", index=False)

print("Number of unique wells:", len(wells))
print("Number of unique scans:", np.sum(scans))

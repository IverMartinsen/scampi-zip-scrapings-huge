import os
import glob
import pandas as pd
import numpy as np

cluster_stats = pd.read_csv("./clusters/cluster_stats.csv", sep=";")
cluster_stats['clean'] = (cluster_stats['black'] != True) * (cluster_stats['blur'] != True) * (cluster_stats['artifact'] != True) * (cluster_stats['multi'] != True)
clean_clusters = cluster_stats[cluster_stats['clean'] == True].index.values
clean_clusters = ['cluster_' + str(i) + '_well_distribution.csv' for i in clean_clusters]

path = '/Users/ima029/Desktop/zip scrapings (huge)/clusters/well_statistics'
paths = [os.path.join(path, f) for f in clean_clusters]

dataframes = [pd.read_csv(p, index_col=0) for p in paths]
# merge the dataframes by index
df = pd.concat(dataframes, axis=1).sum(axis=1)
# change dtype to int
# set the columns name
df = df.astype(int)
df = df.reset_index()
df.columns = ['well', 'count']
# order by well
df = df.sort_values('well')

df.to_csv('curated_well_stats.csv', index=False)

x = df['count']

import matplotlib.pyplot as plt
plt.figure()
plt.hist(x, bins=30, edgecolor='white')
plt.title(f"Curated well distribution")
plt.xlabel("Count", fontsize=8)
plt.xticks(fontsize=8)
plt.ylabel("Frequency", fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig("curated_well_distribution_hist.png", dpi=300)
plt.close()

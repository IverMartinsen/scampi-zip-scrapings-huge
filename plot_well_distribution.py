import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("wells.csv")
wells, counts, scans = df["well"], df["count"], df["scans"]

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
plt.bar(wells[:114], counts[:114] / scans[:114])
plt.xticks(rotation=90, fontsize=2)
plt.yticks(fontsize=4)
plt.ylim(0, 35000)
plt.savefig("wells_part1.png", dpi=600)
plt.close()

plt.figure()
plt.bar(wells[114:228], counts[114:228] / scans[114:228])
plt.xticks(rotation=90, fontsize=2)
plt.yticks(fontsize=4)
plt.ylim(0, 35000)
plt.savefig("wells_part2.png", dpi=600)
plt.close()

plt.figure()
plt.bar(wells[228:], counts[228:] / scans[228:])
plt.xticks(rotation=90, fontsize=2)
plt.yticks(fontsize=4)
plt.ylim(0, 35000)
plt.savefig("wells_part3.png", dpi=600)
plt.close()

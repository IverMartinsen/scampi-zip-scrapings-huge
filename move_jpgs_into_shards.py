import os
import shutil
import glob

path = "data/zip scrapings (huge)/run_2024-02-08"
num_shards = 32

# list jpg files
files = [os.path.basename(f) for f in glob.glob(path + "/*.jpg")]
num_files = len(files)

print(f"Number of files: {num_files}")

batch_size = num_files // num_shards
# make sure the number of files is a multiple of the number of batches
assert num_files % num_shards == 0, "Number of shards must be a multiple of the number of files"

# move files into shards
for i in range(num_shards):
    os.makedirs(path + "/shard_" + str(i), exist_ok=True)
    for j in range(i, len(files), num_shards):
        shutil.move(path + "/" + files[j], path + "/shard_" + str(i) + "/" + files[j])
        
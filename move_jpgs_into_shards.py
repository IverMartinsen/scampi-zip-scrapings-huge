import os
import glob
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="./run_2024-02-08 curated")
parser.add_argument("--num_shards", type=int, default=32)
args = parser.parse_args()

# list jpg files
files = [os.path.basename(f) for f in glob.glob(args.path + "/*.jpg")]
num_files = len(files)

print(f"Number of files: {num_files}")

batch_size = num_files // args.num_shards
# make sure the number of files is a multiple of the number of batches
assert num_files % args.num_shards == 0, "Number of shards must be a multiple of the number of files"

# move files into shards
for i in range(args.num_shards):
    os.makedirs(args.path + "/shard_" + str(i), exist_ok=True)
    for j in range(i, len(files), args.num_shards):
        shutil.move(args.path + "/" + files[j], args.path + "/shard_" + str(i) + "/" + files[j])
        
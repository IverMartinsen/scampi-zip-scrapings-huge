import os
import glob
import h5py
import time
import torch
import torchvision
import concurrent.futures
from PIL import Image


# list paths to all shards
path_to_shards = "/Users/ima029/Desktop/SCAMPI/Repository/data/zip scrapings (huge)/run_2024-02-08"
folders = glob.glob(os.path.join(path_to_shards, "*"))
folders = [folder for folder in folders if os.path.isdir(folder)]        
folders.sort()
folders = folders[28:]
# destination path for hdf5 files
#path_to_hdf5 = "/Users/ima029/Desktop/SCAMPI/Repository/data/zip scrapings (huge)/run_2024-02-08-hdf5"
path_to_hdf5 = "/Users/ima029/Desktop/SCAMPI/Repository/data/zip scrapings (huge)/run_2024-02-08-hdf5-test"
os.makedirs(path_to_hdf5, exist_ok=True)

#read_fn = lambda file: (file, torchvision.io.read_image(file))
read_fn = lambda file: (file, torchvision.io.read_file(file))

for folder in folders:
    start = time.time()

    print(f"Processing {folder}...")
    files = glob.glob(os.path.join(folder, "*.jpg"))
    print(f"Number of files: {len(files)}")
    # create a hdf5 file for each shard
    output_file = os.path.join(path_to_hdf5, os.path.basename(folder) + ".hdf5")

    file_bytes = {}
    # read files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(read_fn, file) for file in files]
        for future in concurrent.futures.as_completed(futures):
            file, image = future.result()
            #image = torchvision.transforms.Resize((224, 224))(image)
            #bytes = torchvision.io.encode_jpeg(image)
            #file_bytes[file] = bytes.numpy()
            file_bytes[file] = image

    with h5py.File(output_file, 'w') as f:
        for file, bytes in file_bytes.items():
            basename = os.path.basename(file)
            f.create_dataset(basename, data=bytes, dtype='uint8')

    end = time.time()
    print(f"Finished processing {folder}!")
    print(f"Time taken: {end - start}")


def read_fn(bytes):
    image = torch.tensor(bytes) # sequence of bytes
    image = torchvision.io.decode_jpeg(image) # shape: (3, H, W)
    image = image.permute(1, 2, 0) # shape: (H, W, 3)
    return image.numpy()


# test the hdf5 files
if __name__ == "__main__":
    path_to_hdf5 = "/Users/ima029/Desktop/SCAMPI/Repository/data/zip scrapings (huge)/run_2024-02-08-hdf5-test/shard_0.hdf5"
    # read the hdf5 file
    with h5py.File(path_to_hdf5, 'r') as f:
        keys = list(f.keys())

    for i, key in enumerate(keys):
        with h5py.File(path_to_hdf5, 'r') as f:
            image = f[key][()] # fancy indexing to get the value of the dataset
            image = read_fn(image)
            image = Image.fromarray(image)
            image.show()
            if i == 10:
                break

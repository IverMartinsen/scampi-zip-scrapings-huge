import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import os

# hyperparameters
data_dir = '/Users/ima029/Desktop/SCAMPI/Repository/data/zip scrapings (huge)/tfds'
image_shape = (224, 224, 3)
shard_size = 96080
num_shards = 32

# Create a FeaturesDict corresponding to the dataset that has been written in tfrecords format
features = tfds.features.FeaturesDict({
    'image': tfds.features.Image(shape=image_shape),
    'label': tfds.features.ClassLabel(names=['fossil']),
    })

# Necessary metadata to write the dataset
split_infos = [
    tfds.core.SplitInfo(
        name='train',
        shard_lengths=[shard_size]*num_shards,  # Num of examples in shard0, shard1,...
        num_bytes=0,  # Total size of your dataset (if unknown, set to 0),
    ),
]

tfds.folder_dataset.write_metadata(
    data_dir=data_dir,
    features=features,
    split_infos=split_infos,
    filename_template=None, # Pass a custom file name template or use None for the default TFDS file name template.
    supervised_keys=('image', 'label'),
    # Optionally, additional DatasetInfo metadata can be provided. See:
    # https://www.tensorflow.org/datasets/api_docs/python/tfds/core/DatasetInfo
    # For example:
    # description="""Multi-line description."""
    # homepage='http://my-project.org',
    # citation="""BibTex citation.""",
)

# test the dataset
builder = tfds.builder_from_directory(data_dir)

print(builder.info)

ds = builder.as_dataset(split='train', shuffle_files=True)

for x in ds.take(1):
    plt.imshow(x['image'])
    plt.show()
    break

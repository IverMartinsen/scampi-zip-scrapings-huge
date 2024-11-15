# Description: Display images from tfrecords file to check if the data is loaded correctly.

import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='data', help='input directory')
parser.add_argument('--compression', type=str, default=None, help='compression type')
args = parser.parse_args()

tfrecords = args.source

def _parse_function(example_proto):
    # Define your feature parsing function
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string)
    }
    image = tf.io.parse_single_example(example_proto, image_feature_description)
    # decode image
    if not args.compression:
        image = tf.io.parse_tensor(image['image'], out_type=tf.uint8)
    elif args.compression == 'jpeg':
        image = tf.io.decode_jpeg(image['image'])
    return image

# read tfrecords
dataset = tf.data.TFRecordDataset(tfrecords)
dataset = dataset.map(_parse_function)

for image in dataset.take(10):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

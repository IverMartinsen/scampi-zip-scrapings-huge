import os
import glob
import argparse
import tensorflow as tf
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='data', help='input directory')
parser.add_argument('--dest', type=str, default='data', help='output directory')
parser.add_argument('--size', type=int, default=224, help='image size')
parser.add_argument('--compression', type=str, default=None, help='compression type')
args = parser.parse_args()

if __name__ == '__main__':
    os.makedirs(args.dest, exist_ok=True)
    
    files = glob.glob(os.path.join(args.source, '*.jpg'))
    # sort files
    files.sort()
    files = files[:1000]
    
    print(f'Found {len(files)} files')
    
    basename = args.source.split('/')[-1]
    
    with tf.io.TFRecordWriter(os.path.join(args.dest, basename + '.tfrecords')) as writer:
        for file in tqdm(files):
            # read and store image as compressed jpeg string
            image = tf.io.read_file(file)
            image = tf.image.decode_jpeg(image)
            image = tf.image.resize(image, (args.size, args.size))
            image = tf.cast(image, tf.uint8)
            if not args.compression:
                image = tf.io.serialize_tensor(image)
            elif args.compression == 'jpeg':
                image = tf.io.encode_jpeg(image)
            image = image.numpy()
            # create tf.train.Example object
            image = tf.train.Example(features=tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
            })).SerializeToString()
            writer.write(image)
            
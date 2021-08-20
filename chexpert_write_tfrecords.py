import os
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import time
import numpy as np


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float64_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def write_tfrecord(input_file, output_dir,base_path):
    """ Write the input file to tfrecord
           base_path: list of file-paths for the images (secondo me è dove sta chexpert) ---> dato che noi abbiamo tutte le cartelle all'interno di chexpert, non abbiamo direttamente la lista dei path alle singole immagini, ma dobbiamo fare quel piccolo preprocessing che vediamo subito all'inizio.
           input_file: location of the csv file to be written
           output_dir: location where the tfrecord file must be saved
    """
    dataset = pd.read_csv(input_file)
    dataset['Path'] = dataset['Path'].apply(lambda x: os.path.join(base_path,x))
    img_addrs = dataset['Path'].values
    img_labels = dataset[dataset.columns[5:]].values

    # open the TFRecords file
    writer = tf.io.TFRecordWriter(output_dir)
    start_time = time.time()
    for i in range(len(img_addrs)):
        # print how many images are saved every 1000 images
        if not i % 500:
            print('Data: {}/{}'.format(i, len(img_addrs)))
            sys.stdout.flush()
        img = open(img_addrs[i], 'rb').read()
        labels = list(img_labels[i])
        feature = {'label': _float64_feature(labels),
                   'image': _bytes_feature(tf.compat.as_bytes(img))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    print('Total time: {}'.format(time.time() - start_time))
    writer.close()
    sys.stdout.flush()



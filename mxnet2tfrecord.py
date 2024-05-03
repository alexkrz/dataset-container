import os
from pathlib import Path

import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Tutorial from https://www.tensorflow.org/tutorials/load_data/tfrecord
# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def main(
    mxnet_dir: str = os.environ["DATASET_DIR"] + "TrainDatasets/casia_webface-mxnet",
    tfrecord_fp: str = os.environ["DATASET_DIR"] + "TrainDatasets/casia_webface.tfrecord",
):
    # files = os.listdir(mxnet_dir)
    # print(files)

    path_imgrec = os.path.join(mxnet_dir, "train.rec")
    path_imgidx = os.path.join(mxnet_dir, "train.idx")
    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    if header.flag > 0:
        header0 = (int(header.label[0]), int(header.label[1]))
        imgidx = np.array(range(1, int(header.label[0])))
    else:
        imgidx = np.array(list(imgrec.keys))

    # print(len(imgidx))
    digits_img = len(str(len(imgidx)))
    # digits_label = 5

    # Load images and labels from mxnet to numpy arrays
    writer = tf.io.TFRecordWriter(tfrecord_fp)
    for i in tqdm(range(len(imgidx))):
        # fname = f"{idx:0{digits_img}d}"
        # print(fname)
        s = imgrec.read_idx(i + 1)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if isinstance(label, float):
            label = int(label)
            # label = f"{label:0{digits_label}d}"
        else:
            raise RuntimeError("Unexpected label dtype")
        img_arr = mx.image.imdecode(img).asnumpy()
        img_bytes = img_arr.tobytes()
        tf_example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image": _bytes_feature(img_bytes),
                    "label": _int64_feature(label),
                }
            )
        )
        writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == "__main__":
    main()

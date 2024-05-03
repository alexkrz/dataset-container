import numbers
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np


def read_mxnet(mxnet_dir: str):
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

    print(len(imgidx))

    idx = 1
    s = imgrec.read_idx(idx)
    header, img = mx.recordio.unpack(s)
    label = header.label
    if not isinstance(label, numbers.Number):
        label = label[0]
    label = np.array(label, dtype=np.int64)
    sample = mx.image.imdecode(img).asnumpy()
    print(sample.shape)
    print(label)
    plt.imshow(sample)
    plt.show()


def main(
    mxnet_dir: str = os.environ["DATASET_DIR"] + "TrainDatasets/casia_webface-mxnet",
):
    # files = os.listdir(mxnet_dir)
    # print(files)

    read_mxnet(mxnet_dir)


if __name__ == "__main__":
    main()

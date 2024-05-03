import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
from tqdm import tqdm


def main(
    mxnet_dir: str = os.environ["DATASET_DIR"] + "TrainDatasets/casia_webface-mxnet",
    hdf5_fp: str = os.environ["DATASET_DIR"] + "TrainDatasets/casia_webface.hdf5",
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
    # img_list = []
    # label_list = []
    h5file = h5py.File(hdf5_fp, "w")
    h5group = h5file.create_group("images")
    for i in tqdm(range(len(imgidx))):
        # fname = f"{idx:0{digits_img}d}"
        # print(fname)
        idx = i + 1
        s = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if isinstance(label, float):
            label = int(label)
            # label = f"{label:0{digits_label}d}"
        else:
            raise RuntimeError("Unexpected label dtype")
        img_np = mx.image.imdecode(img).asnumpy()
        dset = h5group.create_dataset(f"{idx:0{digits_img}d}", data=img_np)
        dset.attrs["CLASS"] = "IMAGE"
        dset.attrs["IMAGE_VERSION"] = "1.2"
        dset.attrs["IMAGE_SUBCLASS"] = "IMAGE_TRUECOLOR"
        dset.attrs["IMAGE_INTERLACE"] = "INTERLACE_PIXEL"
        dset.attrs["IMAGE_MINMAXRANGE"] = np.array([0, 255], dtype=np.uint8)
        if i == 10:
            break
    #     img_list.append(img_np)
    #     label_list.append(label)
    # images = np.array(img_list, dtype=np.uint8)
    # labels = np.array(label_list, dtype=np.int64)

    # h5file = h5py.File(hdf5_fp, "w")
    # h5group = h5file.create_group("casia_webface")
    # h5file.create_dataset(
    #     name="images",
    #     shape=images.shape,
    #     data=images,
    # )
    # h5file.create_dataset(
    #     name="labels",
    #     shape=labels.shape,
    #     data=labels,
    # )
    h5file.close()


if __name__ == "__main__":
    main()

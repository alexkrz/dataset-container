import os
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
from tqdm import tqdm


def zip_directory(directory: str, zip_name: str):
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory):
            for file in files:
                zipf.write(
                    os.path.join(root, file),
                    os.path.relpath(os.path.join(root, file), os.path.join(directory, os.pardir)),
                )


def main(
    mxnet_dir: str = os.environ["DATASET_DIR"] + "TrainDatasets/casia_webface-mxnet",
    img_dir: str = os.environ["DATASET_DIR"] + "TrainDatasets/casia_webface-jpeg",
    img_ext: str = ".jpg",
):
    # files = os.listdir(mxnet_dir)
    # print(files)
    out_dir = Path(img_dir)
    if not out_dir.exists():
        out_dir.mkdir()

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
    digits_label = 5

    # Load images and labels from mxnet to numpy arrays
    for i in tqdm(range(len(imgidx))):
        idx = i + 1
        fname = f"{idx:0{digits_img}d}"
        # print(fname)
        s = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if isinstance(label, float):
            label = int(label)
            label = f"{label:0{digits_label}d}"
        else:
            raise RuntimeError("Unexpected label dtype")
        img_arr = mx.image.imdecode(img).asnumpy()
        label_dir = out_dir / label
        if not label_dir.exists():
            label_dir.mkdir()
        plt.imsave(label_dir / f"{fname}{img_ext}", img_arr)

    # Zip directory
    print("Zip directory..")
    zip_directory(str(out_dir), str(out_dir) + ".zip")


if __name__ == "__main__":
    main()

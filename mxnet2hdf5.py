import io
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
from PIL import Image
from tqdm import tqdm


def main(
    mxnet_dir: str = os.environ["DATASET_DIR"] + "TrainDatasets/casia_webface",
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
    digits_label = 5

    # Load images and labels from mxnet to numpy arrays
    h5file = h5py.File(hdf5_fp, "w")
    # image_dataset = h5file.create_dataset(
    #     name="images",
    #     shape=(len(imgidx), 112, 112, 3),
    #     dtype=np.uint8,
    # )
    # label_dataset = h5file.create_dataset(
    #     name="labels",
    #     shape=(len(imgidx),),
    #     dtype=np.int64,
    # )
    # h5group = h5file.create_group("images")
    labels = []
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

        if label not in labels:
            labels.append(label)
            h5group = h5file.create_group(f"{label:0{digits_label}d}")

        img_arr = mx.image.imdecode(img).asnumpy()
        # img_bytes = img_arr.tobytes()

        # Convert the NumPy array to a PIL Image
        pil_image = Image.fromarray(img_arr)

        # Create a BytesIO object to hold the JPEG bytes
        jpeg_buffer = io.BytesIO()

        # Save the PIL Image as a JPEG into the BytesIO object
        pil_image.save(jpeg_buffer, format="JPEG")

        # Get the JPEG bytes
        jpeg_bytes = jpeg_buffer.getvalue()
        binary_data_np = np.asarray(jpeg_bytes)

        # NOTE: Storing the raw img_arr instead of binary_data_np (with jpeg compression) results in ~10x the file size
        dset = h5group.create_dataset(f"{idx:0{digits_img}d}", data=binary_data_np)

        # image_dataset[i] = img_arr
        # label_dataset[i] = label

    h5file.close()


if __name__ == "__main__":
    main()

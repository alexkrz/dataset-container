import io
import os
import pickle
from pathlib import Path

import lmdb
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
from PIL import Image
from tqdm import tqdm


def main(
    mxnet_dir: str = os.environ["DATASET_DIR"] + "TrainDatasets/casia_webface",
    lmdb_path: str = os.environ["DATASET_DIR"] + "TrainDatasets/casia_webface-lmdb",
    img_ext: str = ".jpg",
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

    # map_size is the maximum size of the database, adjust according to your data
    env = lmdb.open(lmdb_path, map_size=1024**4)

    # Load images and labels from mxnet to numpy arrays
    index_list = []
    with env.begin(write=True) as txn:
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
            # img_bytes = img_arr.tobytes()

            # Convert the NumPy array to a PIL Image
            pil_image = Image.fromarray(img_arr)

            # Create a BytesIO object to hold the JPEG bytes
            jpeg_buffer = io.BytesIO()

            # Save the PIL Image as a JPEG into the BytesIO object
            pil_image.save(jpeg_buffer, format="JPEG")

            # Get the JPEG bytes
            jpeg_bytes = jpeg_buffer.getvalue()

            data = {
                "image": jpeg_bytes,
                "label": label,
            }

            # Serialize the data using pickle
            serialized_data = pickle.dumps(data)

            # Save the image bytes
            txn.put(fname.encode("utf-8"), serialized_data)

            # Maintain the index of image filenames
            index_list.append(fname)

            # if idx == 1000:
            #     break

        # Store the index list in LMDB
        txn.put(b"__index__", pickle.dumps(index_list))

    env.close()


if __name__ == "__main__":
    main()

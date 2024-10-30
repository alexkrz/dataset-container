import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
from datasets import Dataset
from jsonargparse import CLI
from PIL import Image


def main(
    mxnet_dir: Path = Path.home() / "Data" / "TrainDatasets" / "ms1mv3-mxnet",
    out_dir: Path = Path.home() / "Data" / "TrainDatasets" / "parquet-files",
    fname: str = "ms1mv3.parquet",
    # num_shards: int = 20,
):
    assert mxnet_dir.exists()

    def entry_for_id(idx):
        # Read sample at idx
        s = imgrec.read_idx(idx)
        header, img_bytes = mx.recordio.unpack(s)

        # Read image bytes as Pillow image
        pil_image = Image.open(io.BytesIO(img_bytes))

        # Treat label info
        label = header.label
        if isinstance(label, (float, int)):
            return {
                "img": pil_image,
                "label": int(label),
            }
        elif isinstance(label, np.ndarray):
            if len(label) == 2:
                identity = int(label[0])
                # TODO: For BUPT dataset: label[1] is always 1. Where is ethnicity encoded?
                ethnicity = int(label[1])
                return {
                    "img": pil_image,
                    "label": identity,
                }
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def generate_entries():
        for i in range(len(imgidx)):
            idx = i + 1
            yield entry_for_id(idx)

    path_imgrec = str(mxnet_dir / "train.rec")
    path_imgidx = str(mxnet_dir / "train.idx")
    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    if header.flag > 0:
        header0 = (int(header.label[0]), int(header.label[1]))
        imgidx = np.array(range(1, int(header.label[0])))
    else:
        imgidx = np.array(list(imgrec.keys))

    print("Number of images:", len(imgidx))

    ds = Dataset.from_generator(generate_entries)

    # for index in range(num_shards):
    #     shard = ds.shard(index=index, num_shards=num_shards, contiguous=True)
    #     shard.to_parquet(out_dir / f"train-{(index+1):0{5}d}-of-{num_shards:0{5}d}.parquet")

    # NOTE: It seems like the datasets library does the sharding automatically in the cache
    # Therefore it should be fine to just save a single .parquet file
    ds.to_parquet(out_dir / fname)


if __name__ == "__main__":
    CLI(main, as_positional=False)

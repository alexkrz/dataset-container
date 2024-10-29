import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
from datasets import Dataset
from PIL import Image


def main(
    mxnet_dir: Path = Path.home() / "Data" / "TrainDatasets" / "casia_webface-mxnet",
    out_dir: Path = Path.home() / "Data" / "TrainDatasets" / "casia_webface-parquet",
):

    assert mxnet_dir.exists()

    def entry_for_id(imgrec, idx):
        s = imgrec.read_idx(idx)
        header, img_bytes = mx.recordio.unpack(s)
        label = header.label
        if isinstance(label, float):
            label = int(label)
        else:
            raise RuntimeError("Unexpected label dtype")
        # img_arr = mx.image.imdecode(img).asnumpy()
        pil_image = Image.open(io.BytesIO(img_bytes))
        # plt.imshow(pil_image)
        # plt.xlabel(label)
        # plt.show()
        return {
            "img": pil_image,
            "label": label,
        }

    def generate_entries():
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

        for i in range(len(imgidx)):
            idx = i + 1
            yield entry_for_id(imgrec, idx)

    ds = Dataset.from_generator(generate_entries)
    num_shards = 3
    for index in range(num_shards):
        shard = ds.shard(index=index, num_shards=num_shards, contiguous=True)
        shard.to_parquet(out_dir / f"train-{(index+1):0{5}d}-of-{num_shards:0{5}d}.parquet")


if __name__ == "__main__":
    main()

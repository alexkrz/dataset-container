import io
import os
from pathlib import Path

import datasets
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# from renumics import spotlight


def main(
    data_p: str = os.environ["HOME"] + "/Data/Face-Recognition/TrainDatasets/parquet-files/casia_webface.parquet",
):
    assert Path(data_p).exists()
    ds = datasets.Dataset.from_parquet(data_p)
    print("Number of items:", len(ds))

    img, label = ds[0].values()
    print("Image:", img)
    print("Label:", label)

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_xlabel(label)
    # plt.savefig("plot.png")

    # spotlight.show(df)


if __name__ == "__main__":
    main()

import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from PIL import Image

# from renumics import spotlight


def main(data_p: Path = Path.home() / "Data" / "TrainDatasets" / "casia_webface-parquet"):
    assert data_p.exists()
    ds = load_dataset("parquet", data_dir=data_p, split="train")
    print("Number of items:", len(ds))

    img, label = ds[0].values()
    plt.imshow(img)
    plt.xlabel(label)
    plt.show()

    # spotlight.show(df)


if __name__ == "__main__":
    main()

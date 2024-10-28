import io
import os
from pathlib import Path

import datasets
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow
from PIL import Image

# from renumics import spotlight


def main(data_p: Path = Path.home() / "Data" / "CV-Classics" / "cifar-10-parquet"):
    df = pd.read_parquet(data_p / "test-00000-of-00001.parquet")
    print(df.head())
    print(df.dtypes)
    img_bytes = df.loc[0]["img"]["bytes"]
    img = Image.open(io.BytesIO(img_bytes))
    plt.imshow(img)
    plt.show()

    # spotlight.show(df)


if __name__ == "__main__":
    main()

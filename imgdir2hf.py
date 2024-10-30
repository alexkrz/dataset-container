import io
from pathlib import Path

import datasets
import pandas as pd
from datasets import Dataset
from PIL import Image
from tqdm import tqdm


def main(
    data_dir: Path = Path.home() / "Data" / "EmotionRecognition" / "AffectNet",
    out_dir: Path = Path.home() / "Data" / "EmotionRecognition" / "AffectNet-parquet",
    split: str = "test",
):
    # ds = datasets.load_dataset("imagefolder", data_dir=str(data_dir / f"{split}"), split="train")
    # print(ds)
    # ds.to_parquet(out_dir / f"{split}.parquet")
    data_dir = data_dir / split
    subdirs = sorted(list(data_dir.glob("*")))
    entries = []
    for subdir in subdirs:
        label = subdir.name
        label = int(label)
        print("Label:", label)
        img_paths = sorted(list(subdir.glob("*.jpg")))
        for img_p in tqdm(img_paths):
            pil_img = Image.open(img_p)
            jpeg_buffer = io.BytesIO()
            pil_img.save(jpeg_buffer, format="JPEG")
            img_bytes = jpeg_buffer.getvalue()
            entry = {"img": img_bytes, "label": label}
            entries.append(entry)
    df = pd.DataFrame(entries)
    # df = df.sort_values(by="label", ignore_index=True)
    print(df.head())
    print(len(df))

    # TODO: Add label dictionary
    ds = Dataset.from_pandas(df)
    ds = ds.cast_column("img", datasets.Image())

    ds.to_parquet(out_dir / f"{split}.parquet")


if __name__ == "__main__":
    main()

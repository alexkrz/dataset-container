from pathlib import Path

import datasets


def main(
    data_dir: Path = Path.home() / "Data" / "EmotionRecognition" / "AffectNet",
    out_dir: Path = Path.home() / "Data" / "EmotionRecognition" / "AffectNet-parquet",
    split: str = "train",
):
    ds = datasets.load_dataset("imagefolder", data_dir=str(data_dir / f"{split}"), split="train")
    print(ds)
    ds.to_parquet(out_dir / f"{split}.parquet")


if __name__ == "__main__":
    main()

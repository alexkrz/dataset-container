# Dataset Container

Try different dataset formats mentioned in <https://realpython.com/storing-images-in-python/>.

## Setup

We recommend [miniforge](https://conda-forge.org/download/) to set up your python environment.
In case VSCode does not detect your conda environments, install [nb_conda](https://github.com/conda-forge/nb_conda-feedstock) in the base environment.

```bash
conda env create -n $YOUR_ENV_NAME -f environment.yml
conda activate $YOUR_ENV_NAME
pip install -r requirements.txt
pre-commit install
```

## Overview of dataset formats

The initial goal of this repository was to convert face image datasets provided as [MXNet RecordIO](https://mxnet.apache.org/versions/1.7/api/python/docs/api/mxnet/recordio/index.html) file format, as this file format is not actively maintained anymore.

Alternative file formates include:

- HDF5
- LMDB
- Apache Parquet

Due to its popularity in the Huggingface [datasets](https://huggingface.co/docs/datasets/index) library, we recommend to choose the Apache Parquet file format.

For simplicity, we further refer to the Apache Parquet format as hf (huggingface) file format.

## Convert from mxnet to hf

Here we provide an example how to comvert the Casia Webface dataset obtained from the [Insightface repository](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_) from mxnet to hf.

Therefore, we simply run the following script:

```bash
python mxnet2hf.py --config configs/hf_casia_webface.yaml
```

## Convert from image directory to hf

Here we provide an example how to convert a directory of images to the huggingface datasets format.

As dataset example we use the [B3FD dataset](https://github.com/kbesenic/B3FD?tab=readme-ov-file).

To perform the conversion, simply run the following script:

```bash
python imgdir2hf.py --config configs/hf_b3fd.yaml
```

## Inspect hf datasets

To be able to inspect the converted hf datasets, we recommend to checkout the accompanying repository: <https://github.com/alexkrz/datasets-spotlight>

## Todos

- [x] Currently empty

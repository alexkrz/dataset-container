import io
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main(
    hdf5_fp: str = os.environ["DATASET_DIR"] + "TrainDatasets/casia_webface.hdf5",
):
    # List all groups.
    group = []

    # Store all the full data paths (group/file). These are the keys to access the image data.
    data = []

    # Function to recursively store all the keys
    def func(name, obj):
        if isinstance(obj, h5py.Dataset):
            data.append(name)
        elif isinstance(obj, h5py.Group):
            group.append(name)

    h5file = h5py.File(hdf5_fp, "r")
    # This operation fills the previously created lists `group` and `data`.
    h5file.visititems(func)

    hf_data = np.array(h5file[data[0]])
    image = Image.open(io.BytesIO(hf_data))
    print(image)
    # print("Image Size:", image.size)
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    main()

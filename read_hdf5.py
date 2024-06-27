import io
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Function to recursively list all groups and items in an HDF5 file
def visit_file(hdf5_file: h5py.File, return_first: bool = False):
    groups = []
    data = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            data.append(name)
            if return_first:
                return True  # Stop visiting file
        elif isinstance(obj, h5py.Group):
            groups.append(name)

    hdf5_file.visititems(visitor)
    return groups, data


def main(
    hdf5_fp: str = os.environ["DATASET_DIR"] + "TrainDatasets/casia_webface.hdf5",
):
    h5file = h5py.File(hdf5_fp, "r")
    groups, data = visit_file(h5file, return_first=True)

    idx = 0
    hf_data = np.array(h5file[data[idx]])
    label = int(data[idx].split("/")[0])
    print("Label:", label)
    image = Image.open(io.BytesIO(hf_data))
    print(image)
    # print("Image Size:", image.size)
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    main()

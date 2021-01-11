import numpy as np
import albumentations as ab
from math import sqrt

from self_supervised_3d_tasks.preprocessing.utils.crop import do_crop_3d


def crop_patches_3d(image, patches_per_side):
    h, w, d, _ = image.shape

    h_grid = h // patches_per_side
    w_grid = w // patches_per_side
    d_grid = d // patches_per_side

    patches = []
    for i in range(patches_per_side):
        for j in range(patches_per_side):
            for k in range(patches_per_side):

                p = do_crop_3d(image,
                            i * h_grid,
                            j * w_grid,
                            k * d_grid,
                            h_grid,
                            w_grid,
                            d_grid)
                patches.append(p)

    return patches

def preprocess_3d(batch, patches_per_side):
    _, w, h, d, _ = batch.shape
    assert w == h and h == d, "accepting only cube volumes"

    volumes_patches = []

    for volume in batch:
        volumes_patches.append(crop_patches_3d(volume, patches_per_side))

    return np.array(volumes_patches), np.array(volumes_patches)

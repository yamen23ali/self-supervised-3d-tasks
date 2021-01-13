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

def resize(x, new_size):
    # x.shape[3] is the number of channels
    new_image = np.zeros((new_size, new_size, new_size, x.shape[3]))

    for z in range(x.shape[2]):
            new_image[:, :, z, :] = np.array(ab.Resize(new_size, new_size)(image=x[:,:,z])["image"])
    return new_image

def random_flip(x):
    # Flip with prob 0.5
    should_flip = np.random.randint(0, 2)
    if should_flip == 1:
        return np.flip(x, 1)

    return x

def rotate_patch_3d(x):

    rotated_patch = []
    # Randomly choose the rotation access
    rot = np.random.randint(1, 10)

    if rot == 1:
        rotated_patch = np.transpose(np.flip(x, 1), (1, 0, 2, 3))  # 90 deg Z
    elif rot == 2:
        rotated_patch = np.flip(np.transpose(x, (1, 0, 2, 3)), 1)  # -90 deg Z
    elif rot == 3:
        rotated_patch = np.flip(x, (0, 1))  # 180 degrees on z axis
    elif rot == 4:
        rotated_patch = np.transpose(np.flip(x, 1), (0, 2, 1, 3))  # 90 deg X
    elif rot == 5:
        rotated_patch = np.flip(np.transpose(x, (0, 2, 1, 3)), 1)  # -90 deg X
    elif rot == 6:
        rotated_patch = np.flip(x, (1, 2))  # 180 degrees on x axis
    elif rot == 7:
        rotated_patch = np.transpose(np.flip(x, 0), (2, 1, 0, 3))  # 90 deg Y
    elif rot == 8:
        rotated_patch = np.transpose(np.flip(x, 0), (2, 1, 0, 3))  # -90 deg Y
    elif rot == 9:
        rotated_patch = np.flip(x, (0, 2))  # 180 degrees on y axis

    return rotated_patch

def crop_and_resize(x):
    # This might become a hyper parameter but for now let's fix it
    # It indicates how many pieces we would divide each side into
    pieces_num = 4

    # All sides have the same size so anyone would do
    side_length = int(x.shape[0] / pieces_num)

    # Select the starting point of the cropping randomly
    start_x = np.random.randint(0, pieces_num)
    start_y = np.random.randint(0, pieces_num)
    start_z = np.random.randint(0, pieces_num)

    cropped_patch = do_crop_3d(x, start_x, start_y, start_z, side_length, side_length, side_length)

    resized_patch = resize(cropped_patch, x.shape[0])

    return random_flip(resized_patch)

def preprocess_3d(batch, patches_per_side):
    _, w, h, d, _ = batch.shape
    assert w == h and h == d, "accepting only cube volumes"

    volumes_patches = []
    augmentations = np.array([rotate_patch_3d, crop_and_resize])
    augmented_volumes_patches = []

    for volume in batch:
        volumes_patches.append(crop_patches_3d(volume, patches_per_side))

    for volume_patches in volumes_patches:
        augmented_volume_patches = []
        for patch in volume_patches:
            patch = np.array(patch)

            selected_indices = np.random.choice(len(augmentations), size=2, replace=False)
            selected_augmentations = augmentations[selected_indices]

            augmented_volume_patches.append(selected_augmentations[0](patch))
            augmented_volume_patches.append(selected_augmentations[1](patch))
            #augmented_volume_patches.append(patch)

        augmented_volumes_patches.append(augmented_volume_patches)

    return np.array(augmented_volumes_patches), np.array([])

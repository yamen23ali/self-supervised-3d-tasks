import sys
import numpy as np
import albumentations as ab
import nibabel as nib
import skimage.transform as skTrans
from math import sqrt
from self_supervised_3d_tasks.preprocessing.utils.crop import do_crop_3d
from scipy.ndimage.filters import gaussian_filter, sobel

thismodule = sys.modules[__name__]

def crop_in_depth(image, patches_in_depth):
    h, w, d, _ = image.shape

    d_grid = d // patches_in_depth

    patches = []
    for i in range(patches_in_depth):
        p = do_crop_3d(image,
                    0,
                    0,
                    i * d_grid,
                    h,
                    w,
                    d_grid)
        patches.append(p)

    return patches

def resize(patch, new_size):
    # image.shape[3] is the number of channels
    new_patch = np.zeros((new_size, new_size, new_size, patch.shape[3]))

    for z in range(patch.shape[2]):
            new_patch[:, :, z, :] = np.array(ab.Resize(new_size, new_size)(image=patch[:,:,z])["image"])
    return new_patch

def resize_with_interpolation(patch, new_size):
    return skTrans.resize(patch, new_size, order=1, preserve_range=True)

def random_flip(patch):
    # Flip with prob 0.5
    should_flip = np.random.randint(0, 2)
    if should_flip == 1:
        return np.flip(patch, 1)

    return patch

def rotate_patch_3d(patch, **kwargs):
    #print('Rotating patch')

    rotated_patch = []
    # Randomly choose the rotation access
    rot = np.random.randint(1, 10)

    if rot == 1:
        rotated_patch = np.transpose(np.flip(patch, 1), (1, 0, 2, 3))  # Rotate 90 deg Z
    elif rot == 2:
        rotated_patch = np.flip(np.transpose(patch, (1, 0, 2, 3)), 1)  # Rotate -90 deg Z
    elif rot == 3:
        rotated_patch = np.flip(patch, (0, 1))  # Rotate 180 degrees on z axis
    elif rot == 4:
        rotated_patch = np.flip(patch, 0)  # Mirror on x
    elif rot == 5:
        rotated_patch = np.flip(patch, 1)  # Mirror on y
    elif rot == 6:
        rotated_patch = np.flip(patch, 2)  # Mirror on Z
    elif rot == 7:
        rotated_patch = np.flip(patch, (0, 1, 2))  # Full miroring
    elif rot == 8:
        rotated_patch = np.flip(patch, (1, 2))  # Rotate 180 degrees on x axis
    elif rot == 9:
        rotated_patch = np.flip(patch, (0, 2))  # Rotate 180 degrees on y axis

    return rotated_patch

def crop_and_resize(patch, alpha=4, **kwargs):
    #print(f'Crop & Resize patch')
    max_crop_length_x = int(patch.shape[0] / alpha)
    max_start_x = (patch.shape[0] - max_crop_length_x) - 2 # To make sure no IndexOutOfBound occurs

    max_crop_length_y = int(patch.shape[1] / alpha)
    max_start_y = (patch.shape[1] - max_crop_length_y) - 2 # To make sure no IndexOutOfBound occurs

    max_crop_length_z = int(patch.shape[2] / alpha)
    max_start_z = (patch.shape[2] - max_crop_length_z) - 2 # To make sure no IndexOutOfBound occurs

    # Select the starting point of the cropping randomly
    start_x = np.random.randint(0, max_start_x)
    start_y = np.random.randint(0, max_start_y)
    start_z = np.random.randint(0, max_start_z)

    # Crop a cubic image, if another form of cropping is required then we need to pass different
    # values instead of the same max_crop_length for all
    cropped_patch = do_crop_3d(patch, start_x, start_y, start_z, max_crop_length_x, max_crop_length_y, max_crop_length_z)


    resized_patch = resize(cropped_patch, patch.shape[0])

    return random_flip(resized_patch)

def crop_and_resize_without_depth(patch, alpha=2, **kwargs):
    #print("Crop and Resize")
    # All sides have the same size so anyone would do
    max_crop_length = int(patch.shape[0] / alpha)
    max_start = (patch.shape[0] - max_crop_length) - 2 # To make sure no IndexOutOfBound occurs

    # Select the starting point of the cropping randomly
    start_x = np.random.randint(0, max_start)
    start_y = np.random.randint(0, max_start)

    # Cropp a cubic image, if another form of cropping is required then we need to pass different
    # values instead of the same max_crop_length for all
    cropped_patch = do_crop_3d(patch, start_x, start_y, 0, max_crop_length, max_crop_length, patch.shape[2])

    resized_patch = resize_with_interpolation(cropped_patch, patch.shape)
    np.save("original", patch)
    np.save("cropped", resized_patch)

    return random_flip(resized_patch)

def add_gaussian_noise(patch, max_mean=0.3, max_sigma=0.1, **kwargs):
    """"
    default max_mean, max_sigma are optimal to keep at least some features in the resulted image
    """
    #print('Add Gaussian Noise')
    mean = np.random.uniform(-max_mean, max_mean)
    sigma = np.random.uniform(0.0, max_sigma)

    return patch + np.random.normal(mean, sigma, patch.shape)

def apply_gaussian_blur(patch, max_sigma=1.0, **kwargs):
    """"
    default max_sigma is optimal to keep at least some features in the resulted image
    """
    #print('Apply Gaussian Blur')
    sigma = np.random.uniform(0.0, max_sigma)

    return gaussian_filter(patch, sigma)

def apply_sobel_filter(patch, **kwargs):
    #print(f'Apply sobel filter')

    dx = sobel(patch, 0)
    dy = sobel(patch, 1)
    dz = sobel(patch, 2)

    magnitued = np.sqrt(dx**2 + dy**2 + dz**2)
    max_magnitued = np.max(magnitued)

    if max_magnitued == 0.0:
        return magnitued
    else:
        return magnitued * (1.0 / max_magnitued)

def adjust_brightness(patch, max_delta=0.125, **kwargs):
    delta = np.random.uniform(-max_delta, max_delta)

    return patch + delta

def adjust_contrast(patch, lower=0.5, upper=1.5, **kwargs):
    contrast_factor = np.random.uniform(lower, upper)
    patch_mean = np.mean(patch)

    return (contrast_factor * (patch - patch_mean)) + patch_mean

def distort_color(patch, **kwargs):
    #print('Distorting patch color')

    # Shuffle to randomize distortions application order
    distortions = [adjust_brightness, adjust_contrast]
    np.random.shuffle(distortions)

    return distortions[1](distortions[0](patch))

def cut_out(patch, alpha=4, **kwargs):
    #print('Apply cutout patch')

    # All sides have the same size so anyone would do
    max_cutout_length_x = int(patch.shape[0] / alpha)
    max_start_x = (patch.shape[0] - max_cutout_length_x) - 2 # To make sure no IndexOutOfBound occurs

    max_cutout_length_y = int(patch.shape[1] / alpha)
    max_start_y = (patch.shape[1] - max_cutout_length_y) - 2 # To make sure no IndexOutOfBound occurs

    max_cutout_length_z = int(patch.shape[2] / alpha)
    max_start_z = (patch.shape[2] - max_cutout_length_z) - 2 # To make sure no IndexOutOfBound occurs

    # Select the starting point of the cropping randomly
    start_x = np.random.randint(0, max_start_x)
    end_x = start_x +  max_cutout_length_x

    start_y = np.random.randint(0, max_start_y)
    end_y = start_y + max_cutout_length_y

    start_z = np.random.randint(0, max_start_z)
    end_z = start_z +  max_cutout_length_z

    cutout_pixels = np.zeros((end_x - start_x, end_y - start_y, end_z - start_z,1))

    cutout_patch = patch.copy()
    cutout_patch[start_x:end_x, start_y:end_y, start_z:end_z, :] =  cutout_pixels

    return cutout_patch

def global_pair(patch, patch_index, volume_index, volumes):
    #print('Get global pair')

    if len(volumes) == 1: return patch

    valid_choices = np.arange(0, len(volumes))
    # Delete the volume of the current patch
    # so we don't end up with identity transformation
    valid_choices = np.delete(valid_choices, volume_index)
    index = np.random.choice(len(valid_choices), size=1, replace=False)[0]
    selected_volume = valid_choices[index]

    return volumes[selected_volume][patch_index]

def keep_original(patch, **kwargs):
    #print('Keeping the original patch')
    return patch

def get_augmentations(augmentations_names):
    return np.array([getattr(thismodule, name) for name in augmentations_names])

def build_sim_mask(patches_positions):
    print(patches_positions.shape)
    arr_len = len(patches_positions)
    mask = np.ones((arr_len, arr_len))

    for i in range(0, arr_len):
        for j in range(i, arr_len):
            if patches_positions[i] == patches_positions[j]:
                mask[i][j] = 0
                mask[j][i] = 0

    return mask

def preprocess_3d_batch_level_loss(batch, patches_in_depth, augmentations_names, files_names):
    _, w, h, d, _ = batch.shape
    #assert w == h and h == d, "accepting only cube volumes"

    volumes = []
    augmented_volumes_patches = []
    patches_positions = []

    # Convert the augmentations names we get from config file to functions
    augmentations = get_augmentations(augmentations_names)

    for volume in batch:
        volumes.append(crop_in_depth(volume, patches_in_depth))

    for volume_index, volume_patches in enumerate(volumes):
        augmented_patches_1 = []
        augmented_patches_2 = []

        # The position of this volume in the original volume
        # since we originally crop the volumes instead of resizing them
        volume_position = files_names[volume_index].split('.')[0].split('_')[-1]

        for patch_index, patch in enumerate(volume_patches):
            patch = np.array(patch)

            selected_indices = np.random.choice(len(augmentations), size=2, replace=False)
            selected_augmentations = augmentations[selected_indices]

            augmented_patches_1.append(
                selected_augmentations[0](
                    patch=patch, patch_index=patch_index,
                    volume_index=volume_index, volumes=volumes
                )
            )

            augmented_patches_2.append(
                selected_augmentations[1](
                    patch=patch, patch_index=patch_index,
                    volume_index=volume_index, volumes=volumes
                )
            )

            patches_positions.append(
                f'{volume_position}_{patch_index}')

        augmented_volume_patches = np.concatenate((augmented_patches_1, augmented_patches_2), axis=0)
        augmented_volumes_patches.append(augmented_volume_patches)

    patches_positions = np.concatenate(
            (patches_positions, patches_positions), axis=0)
    mask = build_sim_mask(np.array(patches_positions))

    return np.array(augmented_volumes_patches), mask

def preprocess_3d_volume_level_loss(batch, patches_in_depth, augmentations_names):
    _, w, h, d, _ = batch.shape
    #assert w == h and h == d, "accepting only cube volumes"

    volumes = []
    augmented_volumes_patches = []

    # Convert the augmentations names we get from config file to functions
    augmentations = get_augmentations(augmentations_names)

    for volume in batch:
        volumes.append(crop_in_depth(volume, patches_in_depth))

    for volume_index, volume_patches in enumerate(volumes):
        augmented_volume_patches = []
        for patch_index, patch in enumerate(volume_patches):
            patch = np.array(patch)

            selected_indices = np.random.choice(len(augmentations), size=2, replace=False)
            selected_augmentations = augmentations[selected_indices]

            augmented_volume_patches.append(
                selected_augmentations[0](
                    patch=patch, patch_index=patch_index,
                    volume_index=volume_index, volumes=volumes
                )
            )
            augmented_volume_patches.append(
                selected_augmentations[1](
                    patch=patch, patch_index=patch_index,
                    volume_index=volume_index, volumes=volumes
                )
            )

        augmented_volumes_patches.append(augmented_volume_patches)

    return np.array(augmented_volumes_patches), np.zeros(len(batch))
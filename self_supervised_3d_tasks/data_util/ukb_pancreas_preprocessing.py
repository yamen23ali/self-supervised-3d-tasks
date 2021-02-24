import glob
import os
import zipfile

import nibabel as nib
import numpy as np
import traceback
import shutil
from pydicom import dcmread
import skimage.transform as skTrans
from self_supervised_3d_tasks.data_util.nifti_utils import read_scan_find_bbox
from self_supervised_3d_tasks.data_util.resize_and_save_nifty import get_cutted_image, get_padded_image, crop_image

def build_volumes(extracted_folder_path, volume_name, destination_path, crop_size = 96, final_dim=(64, 64, 64)):
    dcm_files = glob.glob(extracted_folder_path + "/*.dcm")
    slices_dict = {}
    slices = []

    for dcm_file_path in dcm_files:
        dcm_file = dcmread(dcm_file_path)
        # Take the normalized version because it's more clear
        if 'NORM' in dcm_file.get('ImageType'):
            slices_dict[dcm_file.get('InstanceNumber')] = dcm_file.pixel_array

    # Sort them because we need slices to be in a correct order
    for key in sorted(slices_dict.keys()):
        slices.append(slices_dict[key])

    # Convert into one 3D image
    volume = np.stack(slices, axis=2)

    # Remove non informative pixels by finding a bounding box
    volume, bb = read_scan_find_bbox(volume)

    # Crop volume into smaller volumes to avoid resizing
    volume = get_cutted_image(volume, crop_size)
    volume = get_padded_image(volume, crop_size)
    volume_crops = crop_image(volume, crop_size)

    for i in range(len(volume_crops)):
        volume_crop = skTrans.resize(volume_crops[i], final_dim, order=1, preserve_range=True)
        volume_crop = np.expand_dims(volume_crop, axis=3)
        np.save(f"{destination_path}/{volume_name}_{i}.npy", volume_crop)

    # Clean up to free space
    shutil.rmtree(extracted_folder_path)

def process_archives():
    source_path = "/Users/d070867/netstore/workspace/ukbio/archive"
    extraction_path = "/Users/d070867/netstore/workspace/ukbio/extracted"
    destination_path = "/Users/d070867/netstore/workspace/ukbio/processed"
    files = glob.glob(source_path + "/*.zip", recursive=True)
    count = 0

    for i, file_path in enumerate(files):
        count+=1
        try:
            base = os.path.basename(file_path)
            file_name = os.path.splitext(base)[0]
            zip_file = zipfile.ZipFile(file_path, 'r')

            # Extract the files from the zip into a folder with same name
            extracted_folder_path = f'{extraction_path}/{file_name}'
            zip_file.extractall(path=extracted_folder_path)

            # build volumes from the slices
            build_volumes(
                extracted_folder_path=extracted_folder_path,
                volume_name=file_name,
                destination_path=destination_path)
        except Exception as ex:
            print(ex)

        perc = (float(i) * 100.0) / len(files)
        print(f"{perc:.2f} % done")

if __name__ == "__main__":
    process_archives()

import glob
import os
import zipfile
import math
import nibabel as nib
import numpy as np
import traceback
import shutil
from pydicom import dcmread
import skimage.transform as skTrans
from self_supervised_3d_tasks.data_util.nifti_utils import read_scan_find_bbox

def get_cutted_image(image, crop_size):
    cutted_image_shape = []
    original_shape = list(image.shape)

    for dim_shape in original_shape:
        if (dim_shape % crop_size == 0) or (dim_shape % crop_size > crop_size//2):
            cutted_image_shape.append(dim_shape)
        else:
            wanted_shape = (dim_shape//crop_size)*crop_size
            cutted_image_shape.append(wanted_shape)

    return image[:cutted_image_shape[0], :cutted_image_shape[1], :cutted_image_shape[2]]

def get_padded_image(image, crop_size):
    padded_image_shape = []
    original_shape = list(image.shape)

    for dim_shape in original_shape:
        if dim_shape % crop_size == 0:
            padded_image_shape.append(dim_shape)
        else:
            wanted_shape = ((dim_shape//crop_size)+1)*crop_size
            padded_image_shape.append(wanted_shape)

    padded_image = np.zeros(padded_image_shape)
    padded_image[:original_shape[0], :original_shape[1], :original_shape[2]] = image

    return padded_image

def get_crop_locations(index, crop_size, shape):
    start = index*crop_size
    end = start + crop_size
    diff = end - shape

    if diff > 0:
        start = start - diff
        end = end - diff

    return start, end

def smart_crop_image(image, crop_shape):
    image_crops = []
    label_crops = []
    crops_names = []
    image_shape = image.shape

    x_crops =  math.ceil(image.shape[0] / crop_shape[0])
    y_crops = math.ceil(image.shape[1] / crop_shape[1])
    z_crops = math.ceil(image.shape[2] / crop_shape[2])

    for i in range(x_crops):
        x_start, x_end = get_crop_locations(i, crop_shape[0], image_shape[0])

        for j in range(y_crops):
            y_start, y_end = get_crop_locations(j, crop_shape[1], image_shape[1])

            for k in range(z_crops):
                z_start, z_end = get_crop_locations(k, crop_shape[2], image_shape[2])

                image_crops.append( image[x_start:x_end, y_start:y_end, z_start:z_end] )
                crops_names.append(f'{x_start}{y_start}{z_start}')

    return image_crops, crops_names


def read_and_store_pancreas(files, data_path , lables_path, save_images_path, save_labels_path):
    dim = (128, 128, 128)
    for i, file_name in enumerate(files):
        path_to_image = "{}/{}".format(data_path, file_name)
        path_to_label = "{}/{}".format(lables_path, file_name)

        try:
            img = nib.load(path_to_image)
            img = img.get_fdata()

            label = nib.load(path_to_label)
            label = label.get_fdata()

            img, bb = read_scan_find_bbox(img)
            label = label[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]

            img = skTrans.resize(img, dim, order=1, preserve_range=True)
            label = skTrans.resize(label, dim, order=1, preserve_range=True)

            result = np.expand_dims(img, axis=3)
            label_result = np.expand_dims(label, axis=3)

            file_name = file_name[:file_name.index('.')] + ".npy"
            label_file_name = file_name[:file_name.index('.')] + "_label.npy"
            np.save("{}/{}".format(save_images_path, file_name), result)
            np.save("{}/{}".format(save_labels_path, label_file_name), label_result)

            perc = (float(i) * 100.0) / len(files)
            print(f"{perc:.2f} % done")

        except Exception as e:
            print("Error while loading image {}.".format(path_to_image))
            traceback.print_tb(e.__traceback__)
            continue

def preprocess_and_store_pancreas(files, data_path , lables_path, save_images_path, save_labels_path, crop_shape = (128, 128, 64), resize_dim=64):
    for i, file_name in enumerate(files):
        path_to_image = "{}/{}".format(data_path, file_name)
        path_to_label = "{}/{}".format(lables_path, file_name)

        try:
            img = nib.load(path_to_image)
            img = img.get_fdata()

            label = nib.load(path_to_label)
            label = label.get_fdata()

            img, bb = read_scan_find_bbox(img)
            label = label[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]

            image_crops, crops_names = smart_crop_image(img, crop_shape)
            label_crops, _ = smart_crop_image(label, crop_shape)

            # Don't resize on depth
            final_dim = (resize_dim, resize_dim, crop_shape[2])

            for j in range(len(image_crops)):
                resized_img = skTrans.resize(image_crops[j], final_dim, order=1, preserve_range=True)
                result = np.expand_dims(resized_img, axis=3)

                resized_label = skTrans.resize(label_crops[j], final_dim, order=1, preserve_range=True)
                label_result = np.expand_dims(resized_label, axis=3)

                image_file_name = file_name[:file_name.index('.')] + f"_{crops_names[j]}.npy"
                label_file_name = file_name[:file_name.index('.')] + f"_{crops_names[j]}_label.npy"
                np.save("{}/{}".format(save_images_path, image_file_name), result)
                np.save("{}/{}".format(save_labels_path, label_file_name), label_result)

            perc = (float(i) * 100.0) / len(files)
            print(f"{perc:.2f} % done")

        except Exception as e:
            print("Error while loading image {}.".format(path_to_image))
            print(e)
            traceback.print_tb(e.__traceback__)
            continue

def prepare_pancreas_data():

    training_images_path = "/home/Yamen.Ali/netstore/pancrease_cropped_64/train"
    training_labels_path = "/home/Yamen.Ali/netstore/pancrease_cropped_64/train_labels"
    test_images_path = "/home/Yamen.Ali/netstore/pancrease_cropped_64/test"
    test_labels_path = "/home/Yamen.Ali/netstore/pancrease_cropped_64/test_labels"

    images_path = "/home/Yamen.Ali/netstore/Task07_Pancreas/imagesTr"
    labels_path = "/home/Yamen.Ali/netstore/Task07_Pancreas/labelsTr"
    '''
    training_images_path = "/Users/d070867/netstore/workspace/cpc_pancreas3d/Task07_Pancreas/Task07_Pancreas/result/train"
    training_labels_path = "/Users/d070867/netstore/workspace/cpc_pancreas3d/Task07_Pancreas/Task07_Pancreas/result/train_labels"
    test_images_path = "/Users/d070867/netstore/workspace/cpc_pancreas3d/Task07_Pancreas/Task07_Pancreas/result/test"
    test_labels_path = "/Users/d070867/netstore/workspace/cpc_pancreas3d/Task07_Pancreas/Task07_Pancreas/result/test_labels"

    images_path = "/Users/d070867/netstore/workspace/cpc_pancreas3d/Task07_Pancreas/Task07_Pancreas/imagesTr"
    labels_path = "/Users/d070867/netstore/workspace/cpc_pancreas3d/Task07_Pancreas/Task07_Pancreas/labelsTr"
    '''


    list_files_temp = np.array(os.listdir(images_path))

    test_size = int(0.3 * len(list_files_temp))
    test_files_indices = np.random.choice(len(list_files_temp), size=test_size, replace=False)
    test_files = list_files_temp[test_files_indices]

    train_files = np.delete(list_files_temp, test_files_indices)

    preprocess_and_store_pancreas(train_files, images_path, labels_path, training_images_path, training_labels_path)
    preprocess_and_store_pancreas(test_files, images_path, labels_path, test_images_path, test_labels_path)

    training_images_path = "/home/Yamen.Ali/netstore/pancrease_resized_128/train"
    training_labels_path = "/home/Yamen.Ali/netstore/pancrease_resized_128/train_labels"
    test_images_path = "/home/Yamen.Ali/netstore/pancrease_resized_128/test"
    test_labels_path = "/home/Yamen.Ali/netstore/pancrease_resized_128/test_labels"

    read_and_store_pancreas(train_files, images_path, labels_path, training_images_path, training_labels_path)
    read_and_store_pancreas(test_files, images_path, labels_path, test_images_path, test_labels_path)

if __name__ == "__main__":
    prepare_pancreas_data()

import csv
import functools
import gc
import os
import random
from os.path import expanduser
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import json
from tensorflow.python.keras import Model
from tensorflow.keras.optimizers import Adam
from self_supervised_3d_tasks.finetune import make_scores

from self_supervised_3d_tasks.test_data_backend import CvDataKaggle, StandardDataLoader
from self_supervised_3d_tasks.train import (
    keras_algorithm_list,
)
from self_supervised_3d_tasks.utils.model_utils import (
    apply_prediction_model,
    get_writing_path,
    print_flat_summary)
from self_supervised_3d_tasks.utils.model_utils import init

def predict(
    algorithm="simclr",
    finetuned_model=None,
    prediction_results_path=None,
    dataset_name="pancreas3d",
    batch_size=5,
    clipnorm=1,
    clipvalue=1,
    lr=1e-3,
    scores=[],
    **kwargs):

    algorithm_def = keras_algorithm_list[algorithm].create_instance(**kwargs)

    data_loader = StandardDataLoader(
        dataset_name,
        batch_size,
        algorithm_def,
        **kwargs)
    gen_train, gen_val, x_test, y_test = data_loader.get_dataset(0, 1)

    enc_model = algorithm_def.get_finetuning_model()
    pred_model = apply_prediction_model(
        input_shape=enc_model.outputs[0].shape[1:],
        algorithm_instance=algorithm_def,
        **kwargs)
    outputs = pred_model(enc_model.outputs)
    model = Model(inputs=enc_model.inputs[0], outputs=outputs)
    model.load_weights(finetuned_model)
    model.compile(
        optimizer=Adam(lr=lr, clipnorm=clipnorm, clipvalue=clipvalue),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    print_flat_summary(model)

    repeate = 100
    y_pred = model.predict(x_test, batch_size=batch_size)

    for i in range(0,repeate):
        y_pred = y_pred + model.predict(x_test, batch_size=batch_size)

    y_pred = y_pred / repeate
    scores_f = make_scores(y_test, y_pred, scores)
    print(scores_f)

    #y_pred = np.argmax(y_pred, axis=-1)
    #for i in range(0,y_pred.shape[0]):
    #    np.save(f'{prediction_results_path}/image_{i}_pred.npy', y_pred[i])

def get_hist_per_image(data_dir_train, **kwargs):
    label_dir = data_dir_train + "_labels"
    label_stem="_label"
    files = os.listdir(data_dir_train)

    data = {}

    for file_name in files:
        path_label = Path("{}/{}".format(label_dir, file_name))
        path_label = path_label.with_name(path_label.stem + label_stem).with_suffix(path_label.suffix)
        data_y = np.load(path_label)
        y = np.rint(data_y).astype(np.int)
        labels, y_counts = np.unique(y, return_counts=True)
        data[file_name] = {
            "class0": str(y_counts[0]),
            "class1": str(y_counts[1]),
            "class2": str(y_counts[2])
        }

    with open(f'{data_dir_train}/hist.json', 'w') as outfile:
        json.dump(data, outfile)

def get_hist(data_dir):
    label_dir = data_dir + "_labels"
    label_stem="_label"
    files = os.listdir(data_dir)

    counts = { 0:0, 1:1, 2:2}

    for file_name in files:
        path_label = Path("{}/{}".format(label_dir, file_name))
        path_label = path_label.with_name(path_label.stem + label_stem).with_suffix(path_label.suffix)
        data_y = np.load(path_label)
        y = np.rint(data_y).astype(np.int)
        labels, y_counts = np.unique(y, return_counts=True)
        counts[0]+= y_counts[0]
        counts[1]+= y_counts[1]
        counts[2]+= y_counts[2]

    return counts

def get_hist_all(data_dir_train, data_dir_test, **kwargs):
    train_counts = get_hist(data_dir_train)
    total = train_counts[0] + train_counts[1] + train_counts[2]
    print(f'Train - Class 0 { (train_counts[0]*100)/ total}')
    print(f'Train - Class 1 { (train_counts[1]*100)/ total}')
    print(f'Train - Class 2 { (train_counts[2]*100)/ total}')

    test_counts = get_hist(data_dir_test)
    total = test_counts[0] + test_counts[1] + test_counts[2]
    print(f'Test - Class 0 { (test_counts[0]*100)/ total}')
    print(f'Test - Class 1 { (test_counts[1]*100)/ total}')
    print(f'Test - Class 2 { (test_counts[2]*100)/ total}')

init(predict, "predict")
#init(get_hist_all, "hist")
#init(get_hist_per_image, "hist")

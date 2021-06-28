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

def average_mc_dropout(model, x_test, batch_size, repeate):

    print("Applying MDC average")

    y_pred = model.predict(x_test, batch_size=batch_size)

    for i in range(0,repeate):
        y_pred = y_pred + model.predict(x_test, batch_size=batch_size)

    y_pred = y_pred / repeate

    return y_pred

def majority_mc_dropout(model, x_test, batch_size, repeate):

    print("Applying MDC majority")

    y_pred = model.predict(x_test, batch_size=batch_size)
    classes_num = y_pred.shape[-1]
    majority_predictions = np.zeros(y_pred.shape)
    majority_predictions = majority_predictions.reshape(-1, classes_num)
    rows = np.arange(len(majority_predictions))

    for i in range(0,repeate):
        y_pred = model.predict(x_test, batch_size=batch_size)
        maxes = np.argmax(y_pred, axis=-1).flatten()
        majority_predictions[rows, maxes] = majority_predictions[rows, maxes] + 1

    return majority_predictions.reshape(y_pred.shape)

def weighted_majority_mc_dropout(model, x_test, batch_size, repeate):

    print("Applying MDC weighted majority")

    y_pred = model.predict(x_test, batch_size=batch_size)
    classes_num = y_pred.shape[-1]
    weighted_majority_predictions = np.zeros(y_pred.shape)
    weighted_majority_predictions = weighted_majority_predictions.reshape(-1, classes_num)
    rows = np.arange(len(weighted_majority_predictions))

    for i in range(0,repeate):
        y_pred = model.predict(x_test, batch_size=batch_size)
        maxes = np.argmax(y_pred, axis=-1).flatten()
        weighted_majority_predictions[rows, maxes] = weighted_majority_predictions[rows, maxes] + (maxes + 1)

    return weighted_majority_predictions.reshape(y_pred.shape)

def borda_mc_dropout(model, x_test, batch_size, repeate):

    print("Applying MDC Borda")

    y_pred = model.predict(x_test, batch_size=batch_size)
    classes_num = y_pred.shape[-1]
    borda_predictions = np.zeros(y_pred.shape)
    borda_predictions = borda_predictions.reshape(-1, classes_num)
    rows = np.arange(len(borda_predictions))

    for i in range(0,repeate):
        y_pred = model.predict(x_test, batch_size=batch_size)
        sorted_indices = np.argsort(y_pred, axis=-1)

        for j in range(1,classes_num):
            indices = sorted_indices[:,:,:,:,j].flatten()
            borda_predictions[rows, indices] = borda_predictions[rows, indices] + j

    return borda_predictions.reshape(y_pred.shape)

def union_mc_dropout(model, x_test, batch_size, repeate, union_class):

    print("Applying MDC union")

    y_pred = model.predict(x_test, batch_size=batch_size)
    classes_num = y_pred.shape[-1]
    union_predictions = np.zeros(y_pred.shape)
    union_predictions = union_predictions.reshape(-1, classes_num)
    rows = np.arange(len(union_predictions))

    for i in range(0,repeate):
        y_pred = model.predict(x_test, batch_size=batch_size)
        maxes = np.argmax(y_pred, axis=-1).flatten()
        union_predictions[rows, maxes] = union_predictions[rows, maxes] + 1

    union_class_predicitions = union_predictions[:,union_class]
    union_indices = union_class_predicitions > 0

    for i in range(0,classes_num):
        if i != union_class:
            union_predictions[union_indices, i] = 0

    return union_predictions.reshape(y_pred.shape)

def weighted_probs_mc_dropout(model, x_test, batch_size, repeate, weights):

    print("Applying MDC weighted probs")

    y_pred = model.predict(x_test, batch_size=batch_size)
    classes_num = y_pred.shape[-1]
    classes_num = y_pred.shape[-1]
    weighted_probs_predictions = np.zeros(y_pred.shape)
    weighted_probs_predictions = weighted_probs_predictions.reshape(-1, classes_num)
    rows = np.arange(len(weighted_probs_predictions))

    for i in range(0,repeate):
        y_pred = model.predict(x_test, batch_size=batch_size)

        for j in range(0,classes_num):
            y_pred[:,:,:,:,j] = y_pred[:,:,:,:,j] * weights[j]

        maxes = np.argmax(y_pred, axis=-1).flatten()
        weighted_probs_predictions[rows, maxes] = weighted_probs_predictions[rows, maxes] + 1

    return weighted_probs_predictions.reshape(y_pred.shape)

def get_compiled_model(
    algorithm,
    finetuned_model,
    clipnorm=1,
    clipvalue=1,
    lr=1e-3,
    dropout_downconv=0.0,
    dropout_upconv=0.0,
    **kwargs):

    algorithm_def = keras_algorithm_list[algorithm].create_instance(
        dropout_downconv=dropout_downconv,
        dropout_upconv=dropout_upconv,
        **kwargs)

    enc_model = algorithm_def.get_finetuning_model()
    pred_model = apply_prediction_model(
        input_shape=enc_model.outputs[0].shape[1:],
        algorithm_instance=algorithm_def,
        dropout_downconv=dropout_downconv,
        dropout_upconv=dropout_upconv,
        **kwargs)
    outputs = pred_model(enc_model.outputs)
    model = Model(inputs=enc_model.inputs[0], outputs=outputs)
    model.load_weights(finetuned_model)
    model.compile(
        optimizer=Adam(lr=lr, clipnorm=clipnorm, clipvalue=clipvalue),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return model

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
    mc_dropout_mode=None,
    mc_dropout_repetetions=1000,
    union_class=2,
    prob_weights=(1,3,6),
    **kwargs):

    algorithm_def = keras_algorithm_list[algorithm].create_instance(**kwargs)

    data_loader = StandardDataLoader(
        dataset_name,
        batch_size,
        algorithm_def,
        **kwargs)
    gen_train, gen_val, x_test, y_test = data_loader.get_dataset(0, 1)

    model = get_compiled_model(
        algorithm,
        finetuned_model,
        clipnorm=clipnorm,
        clipvalue=clipvalue,
        lr=lr,
        **kwargs)

    #print_flat_summary(model)

    y_pred = None

    if mc_dropout_mode=='average':
        y_pred = average_mc_dropout(model, x_test, batch_size, mc_dropout_repetetions)
    elif mc_dropout_mode=='majority':
        y_pred = majority_mc_dropout(model, x_test, batch_size, mc_dropout_repetetions)
    elif mc_dropout_mode=='weighted_majority':
        y_pred = weighted_majority_mc_dropout(model, x_test, batch_size, mc_dropout_repetetions)
    elif mc_dropout_mode=='borda':
        y_pred = borda_mc_dropout(model, x_test, batch_size, mc_dropout_repetetions)
    elif mc_dropout_mode=='union':
        y_pred = union_mc_dropout(model, x_test, batch_size, mc_dropout_repetetions, union_class)
    elif mc_dropout_mode=='weighted_probs':
        y_pred = weighted_probs_mc_dropout(model, x_test, batch_size, mc_dropout_repetetions, prob_weights)
    else:
        y_pred = model.predict(x_test, batch_size=batch_size)

    scores_f = make_scores(y_test, y_pred, scores)
    print(scores_f)

    #y_pred = np.argmax(y_pred, axis=-1)
    #for i in range(0,y_pred.shape[0]):
    #    np.save(f'{prediction_results_path}/image_{i}_pred.npy', y_pred[i])

def get_best_model(files):
    models = [file_name  for file_name in files if 'hdf5' in file_name]
    models.sort(reverse=True)
    return models[0]

def get_worst_image_union(algorithm="simclr",
    finetuned_model=None,
    worst_image_union_path=None,
    dataset_name="pancreas3d",
    batch_size=5,
    clipnorm=1,
    clipvalue=1,
    lr=1e-3,
    scores=[],
    mc_dropout_repetetions=1000,
    union_class=2,
    score_index = 4,
    **kwargs):

    # Hold the values to use them later
    dropout_downconv = kwargs['dropout_downconv']
    kwargs.pop('dropout_downconv', None)
    dropout_upconv = kwargs['dropout_upconv']
    kwargs.pop('dropout_upconv', None)

    algorithm_def = keras_algorithm_list[algorithm].create_instance(**kwargs)

    data_loader = StandardDataLoader(
        dataset_name,
        batch_size,
        algorithm_def,
        **kwargs)
    gen_train, gen_val, x_test, y_test = data_loader.get_dataset(0, 1)

    model = get_compiled_model(
        algorithm,
        finetuned_model,
        clipnorm=clipnorm,
        clipvalue=clipvalue,
        lr=lr,
        dropout_downconv=0.0,
        dropout_upconv=0.0,
        **kwargs)

    x_worst, y_worst, y_worst_pred  = [],[],[]
    min_score = 100
    scores_f_min = []

    for x,y in zip(x_test,y_test):
        x = x[np.newaxis,:,:,:,:]
        y = y[np.newaxis,:,:,:,:]
        y_pred = model.predict(x, batch_size=1)
        scores_f = make_scores(y, y_pred, scores)
        print(scores_f[score_index])
        if scores_f[score_index][1] < min_score:
            min_score = scores_f[score_index][1]
            scores_f_min = scores_f
            x_worst = x
            y_worst = y
            y_worst_pred = y_pred

    print(scores_f_min)

    model = get_compiled_model(
        algorithm,
        finetuned_model,
        clipnorm=clipnorm,
        clipvalue=clipvalue,
        lr=lr,
        dropout_downconv=dropout_downconv,
        dropout_upconv=dropout_upconv,
        **kwargs)

    #y_pred = union_mc_dropout(model, x_worst, 1, mc_dropout_repetetions, union_class)
    y_pred = majority_mc_dropout(model, x_worst, 1, mc_dropout_repetetions)
    scores_f = make_scores(y_worst, y_pred, scores)
    print(scores_f)

    np.save(f'{worst_image_union_path}/image.npy', x_worst)
    np.save(f'{worst_image_union_path}/label.npy', y_worst)
    np.save(f'{worst_image_union_path}/pred.npy', y_worst_pred)

def predict_all(
    finetuned_model=None,
    **kwargs):

    for repetition_dir in os.walk(finetuned_model):
        if 'repetition' not in repetition_dir[0]: continue
        print("========================")
        best_model = get_best_model(repetition_dir[2])
        best_model_path = f'{repetition_dir[0]}/{best_model}'
        print(best_model_path)
        predict(finetuned_model=best_model_path, **kwargs)
        print("========================")

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

        dict_count = ['0','0','0','0']

        for i in range(0,4):
            try:
                dict_count[i] = str(y_counts[i])
            except Exception as e:
                print(e)

        data[file_name] = {
            "class0": dict_count[0],
            "class1": dict_count[1],
            "class2": dict_count[2],
            "class3": dict_count[3]
        }

    with open(f'{data_dir_train}/hist.json', 'w') as outfile:
        json.dump(data, outfile)

def get_hist(data_dir):
    label_dir = data_dir + "_labels"
    label_stem="_label"
    files = os.listdir(data_dir)

    counts = { 0:0, 1:0, 2:0, 3:0}

    for file_name in files:
        path_label = Path("{}/{}".format(label_dir, file_name))
        path_label = path_label.with_name(path_label.stem + label_stem).with_suffix(path_label.suffix)
        data_y = np.load(path_label)
        y = np.rint(data_y).astype(np.int)
        labels, y_counts = np.unique(y, return_counts=True)

        for i in range(0, 4):
            try:
                counts[i]+= y_counts[i]
            except Exception as e:
                print(e)

    return counts

def get_hist_all(data_dir_train, data_dir_test, **kwargs):
    train_counts = get_hist(data_dir_train)
    total = 0

    for i in range(0,4):
        total+= train_counts[i]
    for i in range(0,4):
        print(f'Train - Class {i} { (train_counts[i]*100)/ total}')

    test_counts = get_hist(data_dir_test)
    total = 0

    for i in range(0,4):
        total+= test_counts[i]
    for i in range(0,4):
        print(f'Test - Class {i} { (test_counts[i]*100)/ total}')

#init(predict, "predict")
#init(predict_all, "predict_all")
#init(get_hist_all, "hist")
#init(get_hist_per_image, "hist")
init(get_worst_image_union, "get_worst_image_union")

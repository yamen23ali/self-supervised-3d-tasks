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
from tensorflow.python.keras import Model
from tensorflow.keras.optimizers import Adam

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
    **kwargs):

    algorithm_def = keras_algorithm_list[algorithm].create_instance(**kwargs)

    data_loader = StandardDataLoader(
        dataset_name,
        batch_size,
        algorithm_def,
        **kwargs)
    gen_train, gen_val, x_test, y_test = data_loader.get_dataset(0, 1)

    enc_model = algorithm_def.get_finetuning_model(finetuned_model)
    pred_model = apply_prediction_model(
        input_shape=enc_model.outputs[0].shape[1:],
        algorithm_instance=algorithm_def,
        **kwargs)
    outputs = pred_model(enc_model.outputs)
    model = Model(inputs=enc_model.inputs[0], outputs=outputs)
    model.compile(
        optimizer=Adam(lr=lr, clipnorm=clipnorm, clipvalue=clipvalue),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    y_pred = model.predict(x_test, batch_size=batch_size)
    y_pred = np.argmax(y_pred, axis=-1)

    for i in range(0.. y_pred.shape[0]):
        np.save(f'{prediction_results_path}/image_{i}_pred.npy', y_pred[0])


init(predict, "predict")
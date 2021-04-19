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
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import CSVLogger

import self_supervised_3d_tasks.utils.metrics as metrics
from self_supervised_3d_tasks.utils.callbacks import TerminateOnNaN, NaNLossError, LogCSVWithStart
from self_supervised_3d_tasks.utils.metrics import weighted_sum_loss, jaccard_distance, \
    weighted_categorical_crossentropy, weighted_dice_coefficient, weighted_dice_coefficient_loss, \
    weighted_dice_coefficient_per_class, brats_wt_metric, brats_et_metric, brats_tc_metric, \
    enhanced_weighted_dice_coefficient_loss, generalised_dice_loss_3D,\
    yamen_dice_loss_3D
from self_supervised_3d_tasks.test_data_backend import CvDataKaggle, StandardDataLoader
from self_supervised_3d_tasks.train import (
    keras_algorithm_list,
)
from self_supervised_3d_tasks.utils.model_utils import (
    apply_prediction_model,
    get_writing_path,
    print_flat_summary)
from self_supervised_3d_tasks.utils.model_utils import init

def get_score(score_name):
    if score_name == "qw_kappa":
        return metrics.score_kappa
    elif score_name == "bin_accuracy":
        return metrics.score_bin_acc
    elif score_name == "cat_accuracy":
        return metrics.score_cat_acc
    elif score_name == "dice":
        return metrics.score_dice
    elif score_name == "dice_pancreas_0":
        return functools.partial(metrics.score_dice_class, class_to_predict=0)
    elif score_name == "dice_pancreas_1":
        return functools.partial(metrics.score_dice_class, class_to_predict=1)
    elif score_name == "dice_pancreas_2":
        return functools.partial(metrics.score_dice_class, class_to_predict=2)
    elif score_name == "jaccard":
        return metrics.score_jaccard
    elif score_name == "qw_kappa_kaggle":
        return metrics.score_kappa_kaggle
    elif score_name == "cat_acc_kaggle":
        return metrics.score_cat_acc_kaggle
    elif score_name == "brats_wt":
        return metrics.brats_wt
    elif score_name == "brats_tc":
        return metrics.brats_tc
    elif score_name == "brats_et":
        return metrics.brats_et
    else:
        raise ValueError(f"score {score_name} not found")

def make_custom_metrics(metrics):
    metrics = list(metrics)

    if "weighted_dice_coefficient" in metrics:
        metrics.remove("weighted_dice_coefficient")
        metrics.append(weighted_dice_coefficient)
    if "brats_metrics" in metrics:
        metrics.remove("brats_metrics")
        metrics.append(brats_wt_metric)
        metrics.append(brats_tc_metric)
        metrics.append(brats_et_metric)
    if "weighted_dice_coefficient_per_class_pancreas" in metrics:
        metrics.remove("weighted_dice_coefficient_per_class_pancreas")

        def dice_class_0(y_true, y_pred):
            return weighted_dice_coefficient_per_class(y_true, y_pred, class_to_predict=0)

        def dice_class_1(y_true, y_pred):
            return weighted_dice_coefficient_per_class(y_true, y_pred, class_to_predict=1)

        def dice_class_2(y_true, y_pred):
            return weighted_dice_coefficient_per_class(y_true, y_pred, class_to_predict=2)

        metrics.append(dice_class_0)
        metrics.append(dice_class_1)
        metrics.append(dice_class_2)

    return metrics


def make_custom_loss(loss):
    if loss == "weighted_sum_loss":
        loss = weighted_sum_loss()
    elif loss == "jaccard_distance":
        loss = jaccard_distance
    elif loss == "weighted_dice_loss":
        loss = weighted_dice_coefficient_loss
    elif loss == "weighted_categorical_crossentropy":
        loss = weighted_categorical_crossentropy()
    elif loss == "enhanced_weighted_dice_loss":
        loss = enhanced_weighted_dice_coefficient_loss
    elif loss == "generalised_dice_loss_3D":
        loss = generalised_dice_loss_3D
    elif loss == "yamen_dice_loss_3D":
        loss = yamen_dice_loss_3D

    return loss

def get_optimizer(clipnorm, clipvalue, lr):
    if clipnorm is None and clipvalue is None:
        return Adam(lr=lr)
    elif clipnorm is None:
        return Adam(lr=lr, clipvalue=clipvalue)
    else:
        return Adam(lr=lr, clipnorm=clipnorm, clipvalue=clipvalue)


def make_scores(y, y_pred, scores):
    scores_f = []
    for x in scores:
        score = get_score(x)(y, y_pred)
        if score is None:
            continue
        scores_f.append((x, score))
    return scores_f

def get_scores_big_data(model, x_test, y_test, scores, step_size=40):
    steps = int(len(x_test) / step_size)
    scores_groups = {}

    for i in range(0, steps):
        start_indx = i*step_size
        end_indx = start_indx + step_size
        x_test_batch = x_test[start_indx:end_indx]
        y_test_batch = y_test[start_indx:end_indx]

        y_pred = model.predict(x_test_batch, batch_size=step_size)
        scores_f = make_scores(y_test_batch, y_pred, scores)

        for s in scores_f:
            if s[0] not in scores_groups.keys():
                scores_groups[s[0]] = []
            scores_groups[s[0]].append(s[1])
    scores_f = []
    for key in scores_groups:
        scores_f.append((key, np.array(scores_groups[key]).mean()))

    return scores_f

def plot_and_save(history, plots_path):
    metrics_train = ['weighted_dice_coefficient', 'dice_class_0', 'dice_class_1', 'dice_class_2']
    metrics_val = ['val_weighted_dice_coefficient', 'val_dice_class_0', 'val_dice_class_1', 'val_dice_class_2']
    metrics_labels = ['Weighted Dice', 'Dice 0', 'Dice 1', 'Dice 2']
    files_names = ['Weighted_Dice', 'Dice_0', 'Dice_1', 'Dice_2']

    for i in range(0,len(metrics_train)):
        plt.plot(history.history[metrics_train[i]], label=f'Training')
        plt.plot(history.history[metrics_val[i]], label=f'Validation')
        plt.title('Finetuning Dice')
        plt.ylabel(metrics_labels[i])
        plt.xlabel('Epochs')
        plt.legend(loc="upper left")
        plt.savefig(f'{plots_path}/{files_names[i]}.png')
        plt.clf()

def run_single_test(algorithm_def, gen_train, gen_val, load_weights, freeze_weights, x_test, y_test, lr,
                    batch_size, epochs, epochs_warmup, model_checkpoint, scores, loss, metrics, logging_path, kwargs,
                    clipnorm=None, clipvalue=None, model_callback=None, working_dir=None, plots_path=None):
    print(metrics)
    print(loss)

    metrics = make_custom_metrics(metrics)
    loss = make_custom_loss(loss)

    model = None
    enc_model = None
    dec_model = enc_model
    if load_weights:
        print("Loading weights")
        #enc_model = algorithm_def.get_finetuning_model(model_checkpoint)
        enc_model, dec_model = algorithm_def.get_finetuning_model_with_dec(model_checkpoint)
    else:
        print("Using only model architecture")
        #enc_model = algorithm_def.get_finetuning_model()
        enc_model, dec_model = algorithm_def.get_finetuning_model_with_dec()

    pred_model = apply_prediction_model(
            input_shape=dec_model.outputs[-1].shape[1:],
            algorithm_instance=algorithm_def,
            num_classes=3,
            **kwargs)
    enc_dec_outputs = dec_model(enc_model.outputs)
    outputs = pred_model(enc_dec_outputs)
    model = Model(inputs=enc_model.inputs[0], outputs=outputs)
    print_flat_summary(model)
    print(working_dir)

    if epochs > 0:
        callbacks = [TerminateOnNaN()]

        logging_csv = False
        if logging_path is not None:
            logging_csv = True
            logging_path.parent.mkdir(exist_ok=True, parents=True)
            logger_normal = CSVLogger(str(logging_path), append=False)
            logger_after_warmup = LogCSVWithStart(str(logging_path), start_from_epoch=epochs_warmup, append=True)
        if freeze_weights or load_weights:
            enc_model.trainable = False
            dec_model.trainable = False

        if freeze_weights:
            print(("-" * 10) + "LOADING weights, encoder model is completely frozen")
            if logging_csv:
                callbacks.append(logger_normal)
        elif load_weights:
            assert epochs_warmup < epochs, "warmup epochs must be smaller than epochs"

            print(
                ("-" * 10) + "LOADING weights, encoder model is trainable after warm-up"
            )
            print(("-" * 5) + " encoder model is frozen")

            w_callbacks = list(callbacks)
            if logging_csv:
                w_callbacks.append(logger_normal)

            model.compile(optimizer=get_optimizer(clipnorm, clipvalue, lr), loss=loss, metrics=metrics)
            model.fit(
                x=gen_train,
                validation_data=gen_val,
                epochs=epochs_warmup,
                callbacks=w_callbacks,
            )
            epochs = epochs - epochs_warmup

            enc_model.trainable = True
            dec_model.trainable = True
            print(("-" * 5) + " encoder model unfrozen")

            if logging_csv:
                callbacks.append(logger_after_warmup)
        else:
            print(("-" * 10) + "RANDOM weights, encoder model is fully trainable")
            if logging_csv:
                callbacks.append(logger_normal)

        if working_dir is not None:
            save_checkpoint_every_n_epochs = 5
            mc_c = tf.keras.callbacks.ModelCheckpoint(str(working_dir / "weights-improvement-{epoch:03d}.hdf5"),
                                                      monitor="val_loss",
                                                      mode="min", save_best_only=True)  # reduce storage space
            mc_c_epochs = tf.keras.callbacks.ModelCheckpoint(str(working_dir / "weights-{epoch:03d}.hdf5"),
                                                             period=save_checkpoint_every_n_epochs)  # reduce storage space
            callbacks.append(mc_c)
            callbacks.append(mc_c_epochs)

        # recompile model
        model.compile(optimizer=get_optimizer(clipnorm, clipvalue, lr), loss=loss, metrics=metrics)
        history = model.fit(
            x=gen_train, validation_data=gen_val, epochs=epochs, callbacks=callbacks
        )


    plot_and_save(history, plots_path)

    model.compile(optimizer=get_optimizer(clipnorm, clipvalue, lr), loss=loss, metrics=metrics)
    model.save_weights(f'{plots_path}/finetuned_model.hdf5')

    # To handle big test data without OOM exceptions
    scores_f = []
    if len(x_test) > 10:
        scores_f = get_scores_big_data(
            model=model, x_test=x_test, y_test=y_test,
            scores=scores, step_size=1)
    else:
        y_pred = model.predict(x_test, batch_size=batch_size)
        scores_f = make_scores(y_test, y_pred, scores)

    if model_callback:
        model_callback(model)

    # cleanup
    del pred_model
    del enc_model
    del model

    algorithm_def.purge()
    K.clear_session()

    for i in range(15):
        gc.collect()

    for s in scores_f:
        print("{} score: {}".format(s[0], s[1]))

    return scores_f


def write_result(base_path, row):
    with open(base_path / "results.csv", "a") as csvfile:
        result_writer = csv.writer(csvfile, delimiter=",")
        result_writer.writerow(row)


class MaxTriesExceeded(Exception):
    def __init__(self, func, *args):
        self.func = func
        if args:
            self.max_tries = args[0]

    def __str__(self):
        return f'Maximum amount of tries ({self.max_tries}) exceeded for {self.func}.'


def try_until_no_nan(func, max_tries=4):
    for _ in range(max_tries):
        try:
            return func()
        except NaNLossError:
            print(f"Encountered NaN-Loss in {func}")
    raise MaxTriesExceeded(func, max_tries)


def run_complex_test(
        algorithm,
        dataset_name,
        root_config_file,
        model_checkpoint,
        epochs_initialized=5,
        epochs_random=5,
        epochs_frozen=5,
        repetitions=2,
        batch_size=8,
        exp_splits=(100, 10, 1),
        lr=1e-3,
        epochs_warmup=2,
        scores=("qw_kappa",),
        loss="mse",
        metrics=("mse",),
        clipnorm=None,
        clipvalue=None,
        do_cross_val=False,
        **kwargs,
):
    model_checkpoint = expanduser(model_checkpoint)
    if os.path.isdir(model_checkpoint):
        weight_files = list(Path(model_checkpoint).glob("weights-improvement*.hdf5"))

        if epochs_initialized > 0 or epochs_frozen > 0:
            assert len(weight_files) > 0, "empty directory!"

        weight_files.sort()
        model_checkpoint = str(weight_files[-1])

    kwargs["model_checkpoint"] = model_checkpoint
    kwargs["root_config_file"] = root_config_file
    metrics = list(metrics)

    working_dir = get_writing_path(
        Path(model_checkpoint).expanduser().parent
        / (Path(model_checkpoint).expanduser().stem + "_test"),
        root_config_file,
    )

    algorithm_def = keras_algorithm_list[algorithm].create_instance(**kwargs)

    results = []
    header = ["Train Split"]

    exp_types = []

    if epochs_frozen > 0:
        exp_types.append("Weights_frozen_")

    if epochs_initialized > 0:
        exp_types.append("Weights_initialized_")

    if epochs_random > 0:
        exp_types.append("Weights_random_")

    for exp_type in exp_types:
        for sc in scores:
            for min_avg_max in ["_min", "_avg", "_max"]:
                header.append(exp_type + sc + min_avg_max)

    write_result(working_dir, header)

    if do_cross_val:
        data_loader = CvDataKaggle(dataset_name, batch_size, algorithm_def, n_repetitions=repetitions, **kwargs)
    else:
        data_loader = StandardDataLoader(dataset_name, batch_size, algorithm_def, **kwargs)

    for train_split in exp_splits:
        percentage = 0.01 * train_split
        print("\n--------------------")
        print("running test for: {}%".format(train_split))
        print("--------------------\n")

        a_s = []
        b_s = []
        c_s = []

        for i in range(repetitions):
            logging_base_path = working_dir / "logs"
            plots_path = f'{working_dir}/plots/split_{train_split}/repetition_{i}'
            os.makedirs(plots_path)

            # Use the same seed for all experiments in one repetition
            tf.random.set_seed(i)
            np.random.seed(i)
            random.seed(i)

            gen_train, gen_val, x_test, y_test = data_loader.get_dataset(i, percentage)

            if epochs_frozen > 0:
                logging_a_path = logging_base_path / f"split{train_split}frozen_rep{i}.log"
                a = try_until_no_nan(
                    lambda: run_single_test(algorithm_def, gen_train, gen_val, True, True, x_test, y_test, lr,
                                            batch_size, epochs_frozen, epochs_warmup, model_checkpoint, scores, loss,
                                            metrics,
                                            logging_a_path,
                                            kwargs, clipnorm=clipnorm, clipvalue=clipvalue, plots_path=plots_path))  # frozen
                a_s.append(a)
            if epochs_initialized > 0:
                logging_b_path = logging_base_path / f"split{train_split}initialized_rep{i}.log"
                b = try_until_no_nan(
                    lambda: run_single_test(algorithm_def, gen_train, gen_val, kwargs["load_weights"], False, x_test, y_test, lr,
                                            batch_size, epochs_initialized, epochs_warmup, model_checkpoint, scores,
                                            loss, metrics,
                                            logging_b_path, kwargs, clipnorm=clipnorm, clipvalue=clipvalue, plots_path=plots_path))
                b_s.append(b)
            if epochs_random > 0:
                logging_c_path = logging_base_path / f"split{train_split}random_rep{i}.log"
                c = try_until_no_nan(
                    lambda: run_single_test(algorithm_def, gen_train, gen_val, False, False, x_test, y_test, lr,
                                            batch_size, epochs_random, epochs_warmup, model_checkpoint, scores, loss,
                                            metrics,
                                            logging_c_path,
                                            kwargs, clipnorm=clipnorm, clipvalue=clipvalue,
                                            working_dir=working_dir, plots_path=plots_path))  # random
                c_s.append(c)

        def get_avg_score(list_abc, index):
            sc = [x[index][1] for x in list_abc]
            return np.mean(np.array(sc))

        def get_min_score(list_abc, index):
            sc = [x[index][1] for x in list_abc]
            return np.min(np.array(sc))

        def get_max_score(list_abc, index):
            sc = [x[index][1] for x in list_abc]
            return np.max(np.array(sc))

        scores_a = []
        scores_b = []
        scores_c = []

        for i in range(len(scores)):
            if epochs_frozen > 0:
                scores_a.append(get_min_score(a_s, i))
                scores_a.append(get_avg_score(a_s, i))
                scores_a.append(get_max_score(a_s, i))

            if epochs_initialized > 0:
                scores_b.append(get_min_score(b_s, i))
                scores_b.append(get_avg_score(b_s, i))
                scores_b.append(get_max_score(b_s, i))

            if epochs_random > 0:
                scores_c.append(get_min_score(c_s, i))
                scores_c.append(get_avg_score(c_s, i))
                scores_c.append(get_max_score(c_s, i))

        data = [str(train_split) + "%"]

        if epochs_frozen > 0:
            data += scores_a

        if epochs_initialized > 0:
            data += scores_b

        if epochs_random > 0:
            data += scores_c

        results.append(data)
        write_result(working_dir, data)


def main():
    init(run_complex_test, "test")


if __name__ == "__main__":
    main()

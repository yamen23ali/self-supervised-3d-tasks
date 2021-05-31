import os
import random
import json
import numpy as np

def get_data_generators_internal(data_path, files, data_generator, train_split=None, val_split=None,
                                 train_data_generator_args={},
                                 test_data_generator_args={},
                                 val_data_generator_args={},
                                 **kwargs):
    if val_split:
        assert train_split, "val split cannot be set without train split"

    # Validation set is needed
    if val_split:
        assert val_split + train_split <= 1., "Invalid arguments for splits: {}, {}".format(val_split, train_split)
        # Calculate splits
        train_split = int(len(files) * train_split)
        val_split = int(len(files) * val_split)

        # Create lists
        #print(f'All {files}')
        train = files[0:train_split]
        #print(f'Train {train}')
        val = files[train_split:train_split + val_split]
        #print(f'Val {val}')
        test = files[train_split + val_split:]
        #print(f'Test {test}')

        # create generators
        train_data_generator = data_generator(data_path, train, **train_data_generator_args)
        val_data_generator = data_generator(data_path, val, **val_data_generator_args)

        if len(test) > 0:
            test_data_generator = data_generator(data_path, test, **test_data_generator_args)
            return train_data_generator, val_data_generator, test_data_generator
        else:
            return train_data_generator, val_data_generator, None
    elif train_split:
        assert train_split <= 1., "Invalid arguments for split: {}".format(train_split)

        # Calculate split
        train_split = int(len(files) * train_split)

        # Create lists
        train = files[0:train_split]
        val = files[train_split:]

        # Create data generators
        train_data_generator = data_generator(data_path, train, **train_data_generator_args)

        if len(val) > 0:
            val_data_generator = data_generator(data_path, val, **val_data_generator_args)
            return train_data_generator, val_data_generator
        else:
            return train_data_generator, None
    else:
        train_data_generator = data_generator(data_path, files, **train_data_generator_args)
        return train_data_generator


class CrossValidationDataset():
    def __init__(self, chunks, data_path, data_generator, train_data_generator_args={},
                 test_data_generator_args={},
                 val_data_generator_args={}, **kwargs):

        self.k_fold = len(chunks)
        self.chunks = chunks
        self.data_path = data_path
        self.data_generator = data_generator
        self.kwargs = kwargs

        self.train_data_generator_args = train_data_generator_args
        self.test_data_generator_args = test_data_generator_args
        self.val_data_generator_args = val_data_generator_args

    def make_generators(self, test_chunk, train_split=None, val_split=None):
        test = self.chunks[test_chunk]
        train_val = []
        for i in range(self.k_fold):
            if not (i == test_chunk):
                train_val += self.chunks[i]

        train_and_val = get_data_generators_internal(self.data_path, train_val, self.data_generator,
                                                     train_split=train_split,
                                                     val_split=val_split,
                                                     # val split can only be used to throw away some data
                                                     train_data_generator_args=self.train_data_generator_args,
                                                     val_data_generator_args=self.val_data_generator_args,
                                                     test_data_generator_args=self.test_data_generator_args,
                                                     **self.kwargs)
        test = get_data_generators_internal(self.data_path, test, self.data_generator, train_split=None,
                                            val_split=None,
                                            train_data_generator_args=self.test_data_generator_args,
                                            **self.kwargs)

        if len(train_and_val) > 2:
            train_and_val = train_and_val[:2]  # remove the test generator

        return train_and_val + (test,)


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]


def make_cross_validation(data_path, data_generator, k_fold=5, files=None,
                          train_data_generator_args={},
                          test_data_generator_args={},
                          val_data_generator_args={},
                          shuffle_before_split=False,
                          **kwargs):
    if files is None:
        # List images in directory
        files = os.listdir(data_path)

    if shuffle_before_split:
        random.shuffle(files)

    chunks = chunkify(files, k_fold)
    return CrossValidationDataset(chunks, data_path, data_generator, train_data_generator_args,
                                  test_data_generator_args, val_data_generator_args, **kwargs)


def ensure_class_dist(data_path, class_distribution_path, files, train_split):
    with open(f'{class_distribution_path}/hist.json') as json_file:
        data = json.load(json_file)

    train_split = int(len(files) * train_split)
    train_files = []

    for i in range(200):
        print(f'Trying to achieve dist, trial {i}=====')

        random.shuffle(files)
        train_files = files[:train_split]
        classes_num = [0,0,0,0]
        
        for file_name in train_files:
            for j in range(0,4):
                classes_num[j] += int(data[file_name][f'class{j}'])

        total = np.sum(classes_num)

        class1_dist = int((classes_num[1]*100 / total)*10)
        class2_dist = int((classes_num[2]*100 / total)*10)
        class3_dist = int((classes_num[3]*100 / total)*100)

        #print(class1_dist)
        #print(class2_dist)
        #print(class3_dist)

        #if (class2_dist >= 20 and class2_dist <= 30) and (class1_dist >= 20 and class1_dist <=30):
        #    return files
        
        if (class1_dist >= 10 and class1_dist <= 20) and (class2_dist >= 20 and class2_dist <=30) and (class3_dist >= 63 and class3_dist <=73):
            return files

    return files

def get_data_generators(data_path, data_generator, train_split=None, val_split=None,
                        train_data_generator_args={},
                        test_data_generator_args={},
                        val_data_generator_args={},
                        shuffle_before_split=False,
                        ensure_class_distribution=False,
                        class_distribution_path=None,
                        **kwargs):
    """
    This function generates the data generator for training, testing and optional validation.
    :param data_path: path to files
    :param data_generator: generator to use, first arguments must be data_path and files
    :param train_split: between 0 and 1, percentage of images used for training
    :param val_split: between 0 and 1, percentage of images used for test, None for no validation set
    :param shuffle_before_split:
    :param train_data_generator_args: Optional arguments for data generator
    :param test_data_generator_args: Optional arguments for data generator
    :param val_data_generator_args: Optional arguments for data generator
    :return: returns data generators
    """

    # List images in directory
    files = os.listdir(data_path)

    if shuffle_before_split:
        random.shuffle(files)

    if ensure_class_distribution and train_split:
        ensure_class_dist(data_path, class_distribution_path, files, train_split)

    return get_data_generators_internal(data_path, files, data_generator, train_split=train_split, val_split=val_split,
                                        train_data_generator_args=train_data_generator_args,
                                        test_data_generator_args=test_data_generator_args,
                                        val_data_generator_args=val_data_generator_args,
                                        **kwargs)

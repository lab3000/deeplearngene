import numpy as np
import os
import pandas as pd
import random


class Clade:
    """Class for loading data and defining, compiling, and fitting a
    a population of keras models for genetic algorithm-driven optimization
    of high-performing model architectures"""

    def __init__(self, config, current_generation=0):
        self.C = config  # save sacred config dict
        self.current_generation = current_generation

        # Create folder for storing pickled dataframes and serialized
        # models
        self.experiment_folder = os.path.expanduser\
            ('~/Experiments/' + self.C['environment'])
        if not os.path.exists(self.experiment_folder):
            os.makedirs(self.experiment_folder)

    def load_data(self):
        '''
        This function is setup to load the reuters news dataset from keras,
        and then split the train set into a train and validation set. The
        function can be changed to load any dataset of interest, and will
        work with the rest of the clade object as long as the test, train,
        and val x and y properties are assigned here.
        '''
        from keras.preprocessing import sequence, text
        from sklearn.model_selection import StratifiedKFold
        from keras.utils.np_utils import to_categorical
        from keras.datasets import reuters

        self.max_words = 10000
        (x_train_val, y_train_val), (x_test, y_test) = reuters.load_data(
            num_words=self.max_words, test_split=0.2)

        print('Vectorizing sequence data...')
        tokenizer = text.Tokenizer(num_words=self.max_words)
        x_train_val = tokenizer.sequences_to_matrix(
            x_train_val, mode='binary')
        self.test_x = tokenizer.sequences_to_matrix(
            x_test, mode='binary')
        print('x_ shape:', x_train_val.shape)

        self.nb_classes = np.max(y_train_val) + 1
        print(self.nb_classes, 'classes')

        print('Converting class vector to binary class matrix '
              '(for use with categorical_crossentropy)')
        y_train_val = to_categorical(y_train_val, self.nb_classes)
        self.test_y = to_categorical(y_test, self.nb_classes)

        # make validation data set--better than using validation_split in
        # keras because that does not perserve class balance; this does
        skf = StratifiedKFold(n_splits=10)
        train_idx = {}
        val_idx = {}
        # just used as a placeholder for .split
        y_temp = y_train_val[:, 0]
        split = 0
        for train_index, val_index in skf.split(x_train_val, y_temp):
            train_idx[str(split)], val_idx[
                str(split)] = train_index, val_index
            split += 1

        self.train_x = x_train_val[train_idx['0']]
        self.train_y = y_train_val[train_idx['0']]
        self.val_x = x_train_val[val_idx['0']]
        self.val_y = y_train_val[val_idx['0']]

    def spawn(self):
        err_str = 'this method must be implemented in a subclass in clades.py!'
        raise NotImplementedError(err_str)

    def breed(self):
        err_str = 'this method must be implemented in a subclass in clades.py!'
        raise NotImplementedError(err_str)

    def seed_models(self):
        err_str = 'this method must be implemented in a subclass in clades.py!'
        raise NotImplementedError(err_str)

    def grow_models(self):
        err_str = 'this method must be implemented in a subclass in clades.py!'
        raise NotImplementedError(err_str)

    def select_parents(self):
        err_str = 'this method must be implemented in a subclass in clades.py!'
        raise NotImplementedError(err_str)

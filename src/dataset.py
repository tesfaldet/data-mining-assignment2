import tensorflow as tf
import numpy as np
from utilities import *


class DataSet(object):

    def __init__(self, train, test, num_folds=2):
        # initializing properties
        self._train = train
        self._test = test
        self._num_folds = num_folds
        self._num_per_fold = self._train['data'].shape[0] / self._num_folds
        self._num_examples = self._num_per_fold * (self._num_folds - 1)
        self._total_num_examples = self._num_per_fold + self._num_examples
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._folds = []

        # split training data and labels into k folds
        for i in range(self._num_folds):
            start = i*self._num_per_fold
            end = (i+1)*self._num_per_fold

            # get validation fold
            val_data_fold = \
                self._train['data'][start:end]
            val_labels_fold = \
                self._train['labels'][start:end]

            # gather the rest of the folds as training data and labels
            train_data = \
                np.concatenate((self._train['data'][:start],
                               self._train['data'][end:]))
            train_labels = \
                np.concatenate((self._train['labels'][:start],
                               self._train['labels'][end:]))

            self._folds.append({'train':
                                {'data': train_data,
                                 'labels': train_labels},
                                'validation':
                                {'data': val_data_fold,
                                 'labels': val_labels_fold}
                                })

    def next_batch(self, batch_size, fold=0):
        train_data = self._folds[fold]['train']['data']
        train_labels = self._folds[fold]['train']['labels']

        # sampling with replacement
        indices = np.random.choice(self._num_examples, batch_size)
        X_batch = train_data[indices]
        y_batch = train_labels[indices]

        # update epoch count
        assert batch_size <= self._num_examples
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            # start next epoch
            self._index_in_epoch = self._index_in_epoch - self._num_examples

        return X_batch, y_batch


def load_dataset(train_filename, test_filename, num_folds=2):
    def load_data(filename, train=True):
        def theta(x): return 2*np.pi*(x / 7.0)  # 7 is number of weekdays
        if train:
            print 'Loading training dataset...'
        else:
            print 'Loading test dataset..'
        with open(filename) as f:
            data = [map(float, line.rstrip('\n')
                    .replace('check', '1,0,0,0')
                    .replace('credit_card', '0,1,0,0')
                    .replace('debit_card', '0,0,1,0')
                    .replace('debit_note', '0,0,0,1')
                    .replace('Sunday', str(np.cos(theta(1.0))) + ',' +
                             str(np.sin(theta(1.0))))
                    .replace('Monday', str(np.cos(theta(2.0))) + ',' +
                             str(np.sin(theta(2.0))))
                    .replace('Tuesday', str(np.cos(theta(3.0))) + ',' +
                             str(np.sin(theta(3.0))))
                    .replace('Wednesday', str(np.cos(theta(4.0))) + ',' +
                             str(np.sin(theta(4.0))))
                    .replace('Thursday', str(np.cos(theta(5.0))) + ',' +
                             str(np.sin(theta(5.0))))
                    .replace('Friday', str(np.cos(theta(6.0))) + ',' +
                             str(np.sin(theta(6.0))))
                    .replace('Saturday', str(np.cos(theta(7.0))) + ',' +
                             str(np.sin(theta(7.0))))
                    .replace('yes', '1')
                    .replace('no', '0')
                    .split(','))
                    for line in f]

        data = np.array(data).astype('float32')

        # normalize columns 1, 2, 3, 5, and 7 between 0 and 1
        data[:, 1] = normalize(data[:, 1])
        data[:, 2] = normalize(data[:, 2])
        data[:, 3] = normalize(data[:, 3])
        data[:, 5] = normalize(data[:, 5])
        data[:, 7] = normalize(data[:, 7])

        if train:
            return data[:, :-1], data[:, -1].reshape((-1, 1))  # data and label
        else:
            return data  # just data

    train_data, train_labels = load_data(train_filename)
    test_data = load_data(test_filename, train=False)

    return DataSet(train={'data': train_data, 'labels': train_labels},
                   test={'data': test_data}, num_folds=num_folds)

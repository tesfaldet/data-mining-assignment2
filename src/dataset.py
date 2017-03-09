# import tensorflow as tf
import numpy as np
from utilities import *


class DataSet(object):

    def __init__(self, train, test, folds=2):
        self._num_examples = train['data'].shape[0]
        self._validation = validation
        self._epochs_completed = 0
        self._index_in_epoch = 0

        # split training data and labels into k folds
        self._folds = []
        num_per_fold = self._num_examples / folds
        for i in range(folds - 1):
            train_data_fold = self._train['data'] \
                                         [i*num_per_fold:(i+1)*num_per_fold]
            train_labels_fold = self._train['labels'] \
                                           [i*num_per_fold:(i+1)*num_per_fold]
            val_data_fold = self._train['data'] \
                                       [i*num_per_fold:(i+1)*num_per_fold]
            val_labels_fold = self._train['labels'] \
                                         [i*num_per_fold:(i+1)*num_per_fold]
            self._folds.append({'train':
                                {'data': train_data_fold,
                                 'labels': train_labels_fold},
                                'validation':
                                {'data': val_data_fold,
                                 'labels': val_labels_fold}
                                })
        # take care of remaining data (deals with case where num examples is
        # not divisible by folds)
        train_data_fold = self._train['data'][(folds - 1)*num_per_fold:]
        train_labels_fold = self._train['labels'][(folds - 1)*num_per_fold:]
        self._folds.append({'data': train_data_fold,
                            'labels': train_labels_fold})

    def validation_data(self):
        pass

    def next_batch(self, batch_size, fold=0):
        pass


def load_dataset(train_filename, test_filename, folds=2):
    def load_data(filename, train=True):
        def theta(x): return 2*np.pi*(x / 7.0)
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
                   test={'data': test_data}, folds=folds)

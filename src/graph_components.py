import tensorflow as tf
from src.Dataset import load_dataset
from src.QueueRunner import QueueRunner


"""
DATA HANDLING
"""


def data_layer(name, train_filename, test_filename, num_folds,
               batch_size, num_processes):
    with tf.get_default_graph().name_scope(name):
        # load dataset
        d = load_dataset(train_filename, test_filename, num_folds)

        # get training and validation data for all folds
        with tf.device('/cpu:0'):
            input_shape = list(d._train['data'].shape[1:])
            target_shape = list(d._train['labels'].shape[1:])
            folds_data = []
            for fold in range(num_folds):
                queue_runner = QueueRunner(d, input_shape, target_shape,
                                           batch_size, fold, num_processes)
                X, y = queue_runner.get_inputs()
                X_val = tf.stack(d._folds[fold]['validation']['data'])
                y_val = tf.stack(d._folds[fold]['validation']['labels'])
                folds_data.append({'train':
                                   {'data': X, 'labels': y},
                                   'validation':
                                   {'data': X_val, 'labels': y_val},
                                   'queue_runner': queue_runner})
            test_data = tf.stack(d._test['data'])

        return folds_data, test_data, input_shape, target_shape


"""
MATRIX OPS
"""


def fc(name, input_layer, num_hidden_units):
    """
    Fully-connected layer.

    :param name:  Name of graph module.
    :param input_layer:  Tensor (NxF) where N is batch size and F is
                         number of features.
    :param num_hidden_units:  Number of hidden units to use.

    :type name:  String
    :type input_layer:  Tensor
    :type num_hidden_units:  int32

    :return:  Return Tensor (NxH) where N is batch size and H is number of
              hidden units.
    """
    with tf.get_default_graph().name_scope(name):
        # MSRA initialization
        weight_init = tf.contrib.layers \
                        .variance_scaling_initializer()

        bias_init = tf.constant_initializer(0.0)

        # weight decay
        reg = tf.contrib.layers.l2_regularizer(0.5)

        return tf.contrib.layers \
                 .fully_connected(input_layer, num_hidden_units,
                                  activation_fn=None,
                                  weights_initializer=weight_init,
                                  weights_regularizer=reg,
                                  biases_initializer=bias_init)


"""
ACTIVATIONS
"""


def leaky_relu(name, input_layer, alpha=0.01):
    with tf.get_default_graph().name_scope(name):
        return tf.maximum(tf.multiply(input_layer, alpha), input_layer)


def elu(name, input_layer, alpha=1.0):
    with tf.get_default_graph().name_scope(name):
        return tf.where(tf.greater(input_layer, 0.0),
                        input_layer, alpha * (tf.exp(input_layer) - 1.0))


def softmax(name, input_layer, axis=-1):
    with tf.get_default_graph().name_scope(name):
        return tf.nn.softmax(input_layer, dim=axis)


"""
MISC.
"""


def argmax(input_layer, axis=1):
    return tf.to_float(tf.argmax(input_layer, axis=axis))


def max(input_layer, axis=1):
    return tf.reduce_max(input_layer, axis=axis)


def mean(input_layer, axis=None):
    return tf.reduce_mean(input_layer, axis=axis)


def equal(input_layer_A, input_layer_B):
    return tf.equal(input_layer_A, input_layer_B)


def sum(input_layer, axis=None):
    return tf.reduce_sum(input_layer, axis=axis)


def one_hot(input_layer, depth=2):
    return tf.one_hot(tf.to_int32(input_layer), depth=depth)


def accuracy(name, prediction, target):
    with tf.get_default_graph().name_scope(name):
        correct_predictions = equal(prediction, target)
        accuracy = mean(tf.to_float(correct_predictions))
        return accuracy


def num_examples(input_layer):
    return tf.to_float(tf.shape(input_layer)[0])


def predict(name, input_layer):
    with tf.get_default_graph().name_scope(name):
        return argmax(input_layer)


"""
OBJECTIVE FUNCTIONS
"""


def cross_entropy_loss(name, posterior, target):
    """
    Compute the negative log probability (cross-entropy) between posterior and
    target.

    :param name:  Name of graph module.
    :param posterior:  Tensor (Nx2) of softmax outputs where N is batch size
    :param target:  Tensor (Nx1) of target labels (0 or 1) where N is batch
                    size.

    :type name:  String
    :type posterior:  Tensor
    :type target:  Tensor

    :return:  Return Tensor scalar
    """
    with tf.get_default_graph().name_scope(name):
        target_onehot = one_hot(target)  # convert to 2-dim distribution
        cross_entropy = sum(-target_onehot * tf.log(posterior), axis=1)
        loss = sum(cross_entropy)
        return loss / num_examples(posterior)  # average loss over batch


def cost(name, prediction, target, cost_matrix):
    """
    Calculate cost using given cost matrix:

    :param name:  Name of graph module.
    :param prediction:  Tensor (Nx1) of predictions where N is
                        batch size.
    :param target:  Tensor (Nx1) of target labels (0 or 1) where N is batch
                    size.
    :param cost_matrix:  Tensor (2x2) in format:

                              P=0  P=1
                             +----+----+
                        T=0  | 0  | Y  |
                             +----+----+
                        T=1  | X  | 0  |
                             +----+----+,

    where P is predicted label, T is target label, X is cost associated with
    false negative prediction, and Y is cost associated with false positive
    prediction.

    :type name:  String
    :type predicted:  Tensor
    :type target:  Tensor
    :type cost_matrix:  Tensor
    """
    with tf.get_default_graph().name_scope(name):
        true_positive = cost_matrix[1][1]
        false_positive = cost_matrix[0][1]
        false_negative = cost_matrix[1][0]
        true_negative = cost_matrix[0][0]

        costs = true_positive * target * prediction + \
            true_negative * (1 - target) * (1 - prediction) + \
            false_positive * (1 - target) * prediction + \
            false_negative * target * (1 - prediction)

        total_cost = sum(costs)

        return total_cost

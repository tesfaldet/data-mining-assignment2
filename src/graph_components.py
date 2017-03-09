import tensorflow as tf
import numpy as np


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

        reg = tf.contrib.layers.l2_regularizer(0.5)

        return tf.contrib.layers \
                 .fully_connected(input_layer, num_hidden_units,
                                  activation_fn=leaky_relu,
                                  weights_initializer=weight_init,
                                  weights_regularizer=reg,
                                  biases_initializer=bias_init)


"""
ACTIVATIONS
"""


def leaky_relu(input_layer, alpha=0.01):
    return tf.maximum(tf.multiply(input_layer, alpha), input_layer)


def elu(input_layer, alpha=1.0):
    return tf.where(tf.greater(input_layer, 0.0),
                    input_layer, alpha * (tf.exp(input_layer) - 1.0))


def softmax(input_layer, axis=-1):
    return tf.nn.softmax(input_layer, dim=axis)


"""
MISC.
"""


def predict(name, input_layer, axis=1):
    with tf.get_default_graph().name_scope(name):
        posterior = softmax(input_layer)
        return tf.argmax(posterior, axis=axis)


"""
OBJECTIVE FUNCTIONS
"""


def cost(name, posterior, target, cost_matrix):
    """
    Calculate cost using given cost matrix.

    :param name:  Name of graph module.
    :param posterior:  Tensor (Nx1) of softmax activations where N is
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
        true_positive = cost_matrix[1, 1]
        false_positive = cost_matrix[0, 1]
        false_negative = cost_matrix[1, 0]
        true_negative = cost_matrix[0, 0]

        one_hot_target = tf.one_hot([])

        cost = true_positive * tf.log(posterior) * target
import tensorflow as tf
from src.NeuralNetwork import NeuralNetwork


# config
config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True
config_proto.allow_soft_placement = True
config_proto.log_device_placement = False
my_config = {}
my_config['train_filename'] = 'data/train.arff'
my_config['test_filename'] = 'data/test.arff'
my_config['batch_size'] = 100
my_config['lr'] = 1e-8
my_config['iterations'] = 20000
my_config['snapshot_frequency'] = 20000
my_config['print_frequency'] = 500
my_config['validation_frequency'] = 1000
my_config['num_processes'] = 6
my_config['num_folds'] = 10
my_config['gpu'] = 0
my_config['cost_matrix'] = [[0.0, 5.0], [50.0, 0.0]]

net = NeuralNetwork(config={'tf': config_proto,
                            'user': my_config})
net.run_train()

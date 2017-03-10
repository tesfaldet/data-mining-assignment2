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
my_config['batch_size'] = 10
my_config['lr'] = 0.01
my_config['iterations'] = 10000
my_config['snapshot_frequency'] = 5000
my_config['print_frequency'] = 10
my_config['validation_frequency'] = 100
my_config['num_processes'] = 6
my_config['gpu'] = 0

net = NeuralNetwork(config={'tf': config_proto,
                            'user': my_config})
net.run_train()

import tensorflow as tf
import numpy as np
import os


def normalize(input, a=0, b=1):
    return ((input - np.min(input)) * (b - a)) / \
           (np.max(input) - np.min(input))


def check_snapshots(snapshots_folder='snapshots/', logs_folder='logs/'):
    checkpoint = tf.train.latest_checkpoint(snapshots_folder)

    resume = False
    start_iteration = 0

    if checkpoint:
        choice = ''
        while choice != 'y' and choice != 'n':
            print 'Snapshot file detected (' + checkpoint + \
                  ') would you like to resume? (y/n)'
            choice = raw_input().lower()

            if choice == 'y':
                resume = checkpoint
                start_iteration = int(checkpoint.split(snapshots_folder)
                                      [1][5:-5])
                print 'resuming from iteration ' + str(start_iteration)
            else:
                print 'removing old snapshots and logs, training from scratch'
                resume = False
                for file in os.listdir(snapshots_folder):
                    if file == '.gitignore':
                        continue
                    os.remove(snapshots_folder + file)
                for file in os.listdir(logs_folder):
                    if file == '.gitignore':
                        continue
                    os.remove(logs_folder + file)
    else:
        print "No snapshots found, training from scratch"

    return resume, start_iteration


def load_graph(frozen_graph_filename, name=None, input_map=None):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a
    # graph_def into the current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=input_map,
            return_elements=None,
            name=name,
            op_dict=None,
            producer_op_list=None
        )
    return graph

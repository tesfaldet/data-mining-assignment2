import tensorflow as tf
from src.graph_components import *
import time
import datetime
import numpy as np


class NeuralNetwork(object):

    def __init__(self, config):
        # import config
        self.user_config = config['user']
        self.tf_config = config['tf']

        self.graph = tf.Graph()
        with tf.device('/gpu:' + str(self.user_config['gpu'])):
            # retrieve training and validation data from all folds and
            # retrieve the test data
            self.folds_data, self.test_data = \
                data_layer('input',
                           self.user_config['train_filename'],
                           self.user_config['test_filename'],
                           self.user_config['num_folds'],
                           self.user_config['batch_size'],
                           self.user_config['num_processes'])

            # build network

            # attach losses

            # attach summaries
            self.attach_summaries('summaries')

    def attach_summaries(self, name):
        with tf.name_scope(name):
            # graph costs
            tf.summary.scalar('training cost', self.train_cost)
            tf.summary.scalar('validation cost', self.val_cost)

            # visualize queue usage
            data_queue = self.queue_runner
            data_queue_capacity = data_queue.batch_size * data_queue.n_threads
            tf.summary.scalar('queue saturation',
                              data_queue.queue.size() / data_queue_capacity)

            # merge summaries
            self.summaries = tf.summary.merge_all()

    def build_network(self, name, input_layer):
        with tf.get_default_graph().name_scope(name):
            # TODO: build network here
            return output

    def run_train(self):
        # for cleanliness
        iterations = self.user_config['iterations']
        lr = self.user_config['lr']
        snapshot_frequency = self.user_config['snapshot_frequency']
        print_frequency = self.user_config['print_frequency']
        validation_frequency = self.user_config['validation_frequency']

        with self.graph.as_default():
            with tf.device('/gpu:' + str(self.user_config['gpu'])):
                optimizer = tf.train.AdamOptimizer(learning_rate=lr)
                train_step = optimizer.minimize(self.train_cost)

            """
            Train over iterations, printing loss at each one
            """
            saver = tf.train.Saver(max_to_keep=0, pad_step_number=16)
            with tf.Session(config=self.tf_config) as sess:

                # check snapshots
                resume, start_iteration = check_snapshots()

                # start summary writers
                summary_writer = tf.summary.FileWriter('logs/train',
                                                       sess.graph)
                summary_writer_val = tf.summary.FileWriter('logs/val')

                # start the tensorflow QueueRunners
                tf.train.start_queue_runners(sess=sess)

                # start the data queue runner's threads
                processes = self.queue_runner.start_processes(sess)

                if resume:
                    saver.restore(sess, resume)
                else:
                    sess.run(tf.global_variables_initializer())

                last_print = time.time()
                for i in range(start_iteration, iterations):
                    # run a train step
                    results = sess.run([train_step, self.loss,
                                        self.summaries])

                    # print training information
                    if (i + 1) % print_frequency == 0:
                        time_diff = time.time() - last_print
                        it_per_sec = print_frequency / time_diff
                        remaining_it = iterations - i
                        eta = remaining_it / it_per_sec
                        print 'Iteration %d: loss: %f lr: %f ' \
                              'iter per/s: %f ETA: %s' \
                              % (i + 1, results[1], base_lr, it_per_sec,
                                 str(datetime.timedelta(seconds=eta)))
                        summary_writer.add_summary(results[2], i + 1)
                        summary_writer.flush()
                        last_print = time.time()

                    # print validation information
                    if (i + 1) % validation_frequency == 0:
                        print 'Validating...'

                        # breaking up large validation data into chunks to
                        # prevent out of memory issues
                        avg_val_loss, val_summary = self.validate_chunks(sess)

                        print 'Validation loss: %f' % (avg_val_loss)
                        summary_writer_val.add_summary(val_summary, i + 1)
                        summary_writer_val.flush()

                    # save snapshot
                    if (i + 1) % snapshot_frequency == 0:
                        print 'Saving snapshot...'
                        saver.save(sess, 'snapshots/iter', global_step=i+1)

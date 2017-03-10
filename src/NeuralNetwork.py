import tensorflow as tf
from src.graph_components import *
from src.utilities import check_snapshots
import time
import datetime


class NeuralNetwork(object):

    def __init__(self, config):
        # import config
        self.user_config = config['user']
        self.tf_config = config['tf']

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device('/gpu:' + str(self.user_config['gpu'])):
                # retrieve training and validation data from all folds and
                # retrieve the test data
                self.folds_data, self.test_data, input_shape, target_shape = \
                    data_layer('data_layer',
                               self.user_config['train_filename'],
                               self.user_config['test_filename'],
                               self.user_config['num_folds'],
                               self.user_config['batch_size'],
                               self.user_config['num_processes'])

                # create input and target placeholders for feeding in training
                # data, validation data, or test data
                self.input_layer = tf.placeholder(dtype=tf.float32,
                                                  shape=[None] + input_shape,
                                                  name='input')
                self.target = tf.placeholder(dtype=tf.float32,
                                             shape=[None] + target_shape,
                                             name='target')

                # build network
                self.output = self.build_network('NeuralNetwork',
                                                 self.input_layer)

                # attach losses
                self.train_cost = cost('train_cost', self.output, self.target,
                                       self.user_config['cost_matrix'],
                                       train=True)
                self.val_cost = cost('validation_cost', self.output,
                                     self.target,
                                     self.user_config['cost_matrix'],
                                     train=False)

                # attach summaries
                self.attach_summaries('summaries')

    def attach_summaries(self, name):
        with tf.get_default_graph().name_scope(name):
            # graph costs
            tf.summary.scalar('training cost', self.train_cost)
            tf.summary.scalar('validation cost', self.val_cost)

            # visualize queue usage TODO: fix
            # data_queue = self.queue_runner
            # data_queue_capacity = data_queue.batch_size * \
            #     data_queue.num_processes
            # tf.summary.scalar('queue saturation',
            #                   data_queue.queue.size() / data_queue_capacity)

            # merge summaries
            self.summaries = tf.summary.merge_all()

    def build_network(self, name, input_layer):
        with tf.get_default_graph().name_scope(name):
            # first hidden layer
            fc1 = fc('fc1', input_layer, 16)

            # first activation
            h_fc1 = elu('elu1', fc1)

            # second hidden layer
            fc2 = fc('fc2', h_fc1, 32)

            # second activation
            h_fc1 = elu('elu2', fc2)

            # output
            output = fc('fc3', h_fc1, 2)

            # activation output
            h_output = softmax('softmax', output)

            return h_output

    def run_train(self):
        # for cleanliness
        iterations = self.user_config['iterations']
        lr = self.user_config['lr']
        snapshot_frequency = self.user_config['snapshot_frequency']
        print_frequency = self.user_config['print_frequency']
        validation_frequency = self.user_config['validation_frequency']
        folds = self.folds_data

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

                # start summary writer
                summary_writer = tf.summary.FileWriter('logs', sess.graph)

                # start the tensorflow QueueRunners
                tf.train.start_queue_runners(sess=sess)

                # start the data queue runner's threads
                # TODO: implement queue stop after fold is done
                fold = 0
                self.queue_runner = folds[fold]['queue_runner']
                processes = self.queue_runner.start_processes(sess)

                if resume:
                    saver.restore(sess, resume)
                else:
                    sess.run(tf.global_variables_initializer())

                # TODO: loop through all folds
                last_print = time.time()
                for i in range(start_iteration, iterations):
                    # retrieve training data
                    input_layer = sess.run(folds[fold]['train']['data'])
                    target = sess.run(folds[fold]['train']['labels'])

                    # run a train step
                    results = sess.run([train_step, self.train_cost,
                                        self.summaries],
                                       feed_dict={self.input_layer:
                                                  input_layer,
                                                  self.target: target})

                    # print training information
                    if (i + 1) % print_frequency == 0:
                        time_diff = time.time() - last_print
                        it_per_sec = print_frequency / time_diff
                        remaining_it = iterations - i
                        eta = remaining_it / it_per_sec
                        print 'Iteration %d: cost: %f lr: %f ' \
                              'iter per/s: %f ETA: %s' \
                              % (i + 1, results[1], lr, it_per_sec,
                                 str(datetime.timedelta(seconds=eta)))
                        summary_writer.add_summary(results[2], i + 1)
                        summary_writer.flush()
                        last_print = time.time()

                    # print validation information
                    if (i + 1) % validation_frequency == 0:
                        print 'Validating...'

                        # retrieve validation data
                        val_input_layer = \
                            sess.run(folds[fold]['validation']['data'])
                        val_target = \
                            sess.run(folds[fold]['validation']['labels'])

                        # evaluate validation cost
                        results = sess.run([self.val_cost, self.summaries],
                                           feed_dict={self.input_layer:
                                                      val_input_layer,
                                                      self.target: val_target})

                        print 'Validation cost: %f' % (results[0])
                        summary_writer.add_summary(results[1], i + 1)
                        summary_writer.flush()

                    # save snapshot
                    if (i + 1) % snapshot_frequency == 0:
                        print 'Saving snapshot...'
                        saver.save(sess, 'snapshots/iter', global_step=i+1)

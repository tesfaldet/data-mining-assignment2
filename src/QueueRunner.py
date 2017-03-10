import tensorflow as tf
from multiprocessing import Process


class QueueRunner(object):
    """
    This class manages the the background processes needed to fill a queue full
    of data.
    """
    def __init__(self, dataset, input_shape, target_shape,
                 batch_size, num_processes=1):
        def flatten(list): return [item for sublist in l for item in sublist]
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_processes = num_processes
        input_shape = flatten([None, flatten(input_shape)])
        target_shape = flatten([None, flatten(target_shape)])
        self.dataX = tf.placeholder(dtype=tf.float32, shape=input_shape)
        self.dataY = tf.placeholder(dtype=tf.float32, shape=target_shape)
        # The actual queue of data.
        self.queue = tf.RandomShuffleQueue(shapes=[input_shape[1:],
                                                   target_shape[1:]],
                                           dtypes=[tf.float32, tf.float32],
                                           capacity=batch_size*num_processes,
                                           min_after_dequeue=batch_size)

        # The symbolic operation to add data to the queue
        self.enqueue_op = self.queue.enqueue_many([self.dataX, self.dataY])

    def get_inputs(self):
        """
        Returns tensors containing a batch of images and labels
        """
        images_batch, labels_batch = self.queue.dequeue_many(self.batch_size)
        return images_batch, labels_batch

    def process(self, sess):
        """
        Function run on alternate process. Basically, keep adding data to the
        queue.
        """
        for dataX, dataY in self._data_iterator():
            sess.run(self.enqueue_op, feed_dict={self.dataX: dataX,
                                                 self.dataY: dataY})

    def start_processes(self, sess):
        """ Start background processes to feed queue """
        processes = []
        for n in range(self.num_processes):
            p = Process(target=self.process, args=(sess,))
            p.daemon = True
            p.start()
            processes.append(p)
        return processes

    def _data_iterator(self):
        while True:
            x_batch, y_batch = self.dataset.next_batch(self.batch_size)
            yield x_batch, y_batch

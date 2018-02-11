# coding: utf-8
from enum import Enum

import tensorflow as tf
from datetime import datetime
import numpy as np


class TaskType(Enum):
    Classification = 'Classification'
    Embedding = 'Embedding'


class TextCNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters,
                 l2_reg_lambda=0.0, task=TaskType.Classification,
                 dropout_keep_prob_value=0.5) -> None:
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.current_learning_rate = 1e-3
        self.previous_gradients = None
        self.dropout_keep_prob_value = dropout_keep_prob_value
        with tf.name_scope('general'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # keeping track of l2 regularization loss
        l2_loss = tf.constant(0.0)

        # word embedding layer
        with tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W')
            self.embedding_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedding_chars_expanded = tf.expand_dims(self.embedding_chars, -1)

        # create a convolution + max-pool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # Convolution layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(self.embedding_chars_expanded, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                # Apply non-linearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name='pool')
                pooled_outputs.append(pooled)

        # combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # add dropout
        with tf.name_scope('drop_out'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # scores and predictions
        with tf.name_scope('drop_out'):
            W = tf.get_variable('W', shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='score')
            if task == TaskType.Classification:
                self.predictions = tf.argmax(self.scores, 1, name='predictions')

        # calculate mean cross-entropy loss
        with tf.name_scope('loss'):
            if task == TaskType.Classification:
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            else:
                losses = tf.losses.mean_squared_error(predictions=tf.tanh(self.scores), labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # calculate accuracy
        with tf.name_scope("accuracy"):
            if task == TaskType.Classification:
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

    def _train_step(self, x_batch, y_batch, session, train_op, gradients):
        feed_dict = {self.input_x: x_batch,
                     self.input_y: y_batch,
                     self.dropout_keep_prob: self.dropout_keep_prob_value,
                     self.learning_rate: self.current_learning_rate}
        _, step, loss, accuracy, grads = session.run([train_op, self.global_step, self.loss, self.accuracy, gradients], feed_dict)
        self._update_learning_rate(grads)

        time_str = datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}, lr {:g}".format(time_str, step, loss, accuracy, self.current_learning_rate))

    def _test_step(self, x_batch, y_batch, session):
        feed_dict = {self.input_x: x_batch,
                     self.input_y: y_batch,
                     self.dropout_keep_prob: 1.0,
                     self.learning_rate: self.current_learning_rate}
        step, loss, accuracy = session.run([self.global_step, self.loss, self.accuracy], feed_dict)

        time_str = datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

    def _update_learning_rate(self, gradients):
        if self.previous_gradients is not None:
            s = np.sum([np.sum(np.multiply(x, y)) for x, y in zip(gradients, self.previous_gradients)])
            # s = tf.reduce_sum([tf.reduce_sum(tf.multiply(x, y)) for x, y in zip(gradients, self.previous_gradients)])
            s = np.clip(s, -1., 1.)
            self.current_learning_rate += 0.001 * s
            self.current_learning_rate = np.clip(self.current_learning_rate, 1e-8, 1e-2)
        self.previous_gradients = gradients

    def train(self, batches, test_data, session):
        x_test, y_test = zip(*test_data)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        gradients = [tf.convert_to_tensor(g) for g, v in grads_and_vars]

        # Initialize all variables
        session.run(tf.global_variables_initializer())

        for batch in batches:
            x_batch, y_batch = zip(*batch)
            self._train_step(x_batch, y_batch, session, train_op, gradients)
            current_step = tf.train.global_step(session, self.global_step)
            if current_step % 100 == 0:
                print("\nEvaluation:")
                self._test_step(x_test, y_test, session)
                print("")

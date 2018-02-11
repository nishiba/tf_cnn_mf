# coding: utf-8
from enum import Enum

import tensorflow as tf
from datetime import datetime
import numpy as np

from model.text_cnn import TextCNN, TaskType


class ConvMF(object):
    def __init__(self, num_items, num_users, sequence_length,
                 num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters,
                 l2_reg_user_lambda=10.0,
                 l2_reg_item_lambda=100.0,
                 l2_reg_conv_lambda=10.0,
                 dropout_keep_prob_value=0.5) -> None:

        # Placeholders for input, output and dropout
        self.input_item = tf.placeholder(tf.int32, [None, 1], name='input_item')
        self.input_user = tf.placeholder(tf.int32, [None, 1], name='input_user')
        self.input_rating = tf.placeholder(tf.float32, [None, 1], name='input_rating')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.current_learning_rate = 1e-3
        self.previous_gradients = None
        self.dropout_keep_prob_value = dropout_keep_prob_value

        with tf.name_scope('general'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # setup TextCNN
        # with tf.name_scope('text_cnn'):
        #     self.text_cnn = TextCNN(sequence_length=sequence_length, num_classes=num_classes, vocab_size=vocab_size,
        #                             embedding_size=embedding_size, filter_sizes=filter_sizes, num_filters=num_filters,
        #                             l2_reg_lambda=l2_reg_conv_lambda, dropout_keep_prob_value=dropout_keep_prob_value,
        #                             task=TaskType.Embedding)

        # item embedding layer
        scale = 1.0 / np.sqrt(embedding_size)
        with tf.name_scope('item_embedding'):
            self.item_W = tf.Variable(tf.random_uniform([num_items, embedding_size], -scale, scale), name='W')
            self.embedding_items = tf.nn.embedding_lookup(self.item_W, self.input_item)

        # user embedding layer
        with tf.name_scope('user_embedding'):
            self.user_W = tf.Variable(tf.random_uniform([num_users, embedding_size], -scale, scale), name='W')
            self.embedding_users = tf.nn.embedding_lookup(self.user_W, self.input_user)

        # calculate loss and l2 regularization
        with tf.name_scope('loss'):
            self.predictions = tf.reduce_sum(tf.multiply(self.embedding_items, self.embedding_users), axis=2, name='prediction')
            losses = tf.losses.mean_squared_error(predictions=self.predictions, labels=self.input_rating)
            self.l2_user_loss = tf.reduce_mean(tf.square(self.embedding_users))
            self.l2_item_loss = tf.reduce_mean(tf.square(self.embedding_items))
            self.error = tf.sqrt(tf.reduce_mean(losses), name='error')
            self.loss = tf.reduce_mean(losses) + l2_reg_user_lambda * self.l2_user_loss + l2_reg_item_lambda * self.l2_item_loss

    def _train_step(self, item_batch, user_batch, rating_batch, session, train_op, gradients):
        feed_dict = {self.input_item: item_batch,
                     self.input_user: user_batch,
                     self.input_rating: rating_batch,
                     self.learning_rate: self.current_learning_rate}
        _, step, loss, error, grads = session.run([train_op, self.global_step, self.loss, self.error, gradients], feed_dict)
        self._update_learning_rate(grads)

        time_str = datetime.now().isoformat()
        print("{}: step {}, loss {:g}, error {:g}, lr {:g}".format(time_str, step, loss, error, self.current_learning_rate))

    def _test_step(self, item_batch, user_batch, rating_batch, session):
        feed_dict = {self.input_item: item_batch,
                     self.input_user: user_batch,
                     self.input_rating: rating_batch,
                     self.learning_rate: self.current_learning_rate}
        step, loss, error = session.run([self.global_step, self.loss, self.error], feed_dict)

        time_str = datetime.now().isoformat()
        print("{}: step {}, loss {:g}, error {:g}".format(time_str, step, loss, error))

    def _update_learning_rate(self, gradients):
        if self.previous_gradients is not None:
            s = np.sum([np.sum(np.multiply(x, y)) for x, y in zip(gradients, self.previous_gradients)])
            s = np.clip(s, -1., 1.)
            self.current_learning_rate += 0.001 * s
            self.current_learning_rate = np.clip(self.current_learning_rate, 1e-8, 1e-2)
        self.previous_gradients = gradients

    def train(self, batches, test_data, session):
        item_test, user_test, rating_test = zip(*test_data)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        gradients = [tf.convert_to_tensor(g) for g, v in grads_and_vars if g is not None]

        # Initialize all variables
        session.run(tf.global_variables_initializer())

        for batch in batches:
            item_batch, user_batch, rating_batch = zip(*batch)
            self._train_step(item_batch, user_batch, rating_batch, session, train_op, gradients)
            current_step = tf.train.global_step(session, self.global_step)
            if current_step % 100 == 0:
                print("\nEvaluation:")
                self._test_step(item_test, user_test, rating_test, session)
                print("")

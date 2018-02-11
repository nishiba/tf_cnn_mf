# coding: utf-8

import time
from itertools import chain
from typing import List, Tuple, Dict

import tensorflow as tf
from sklearn.model_selection import train_test_split

from model.convmf import ConvMF
from model.text_cnn import TextCNN
import os

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import re


def make_rating_data(size: int = None):
    data = pd.read_csv(os.path.join('data', 'ratings.csv')).rename(columns={'movie': 'item'})
    data.user = data.user.astype(np.int32)
    data.item = data.item.astype(np.int32)
    data.rating = data.rating.astype(np.float32)
    if size is not None:
        data = data.head(size)
    items = data.item.values.reshape((-1, 1))
    users = data.user.values.reshape((-1, 1))
    ratings = data.rating.values.reshape((-1, 1))
    return list(zip(items, users, ratings)), np.max(data.item) + 1, np.max(data.user) + 1


def batch_iter(data, batch_size, num_epochs, do_shuffle=True):
    data_size = len(data)
    print(data_size)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if do_shuffle:
            data = shuffle(data)

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            yield data[start_index:start_index + batch_size]


def train(num_items,
          num_users,
          batches,
          test_data,
          sequence_length,
          num_classes,
          vocab_size,
          embedding_size,
          filter_sizes,
          num_filters,
          l2_reg_lambda):
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=tf.GPUOptions(visible_device_list='0,1', allow_growth=True))
    sess = tf.Session(config=session_conf)
    with tf.device("/gpu:0"), sess.as_default():
        cnn = ConvMF(
            num_items=num_items,
            num_users=num_users,
            sequence_length=sequence_length,
            num_classes=num_classes,
            vocab_size=vocab_size,
            filter_sizes=filter_sizes,
            num_filters=num_filters,
            embedding_size=embedding_size,
            l2_reg_conv_lambda=l2_reg_lambda)

        # Checkpoint directory.
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # Generate batches
        cnn.train(batches, test_data, sess)
        path = saver.save(sess, checkpoint_prefix)
        print("Saved model checkpoint to {}\n".format(path))


def main():
    max_sentence_length = 60
    batch_size = 2 ** 12
    num_epochs = 5

    dataset, num_items, num_users = make_rating_data()
    train_data, test_data = train_test_split(dataset, random_state=123, test_size=0.1)
    train_iter = batch_iter(train_data, batch_size=batch_size, num_epochs=num_epochs, do_shuffle=True)

    num_classes = 2
    embedding_size = 300
    filter_sizes = [3, 4, 5]
    num_filters = 100
    l2_reg_lambda = 10
    vocab_size = 100

    train(num_items, num_users,
          train_iter, test_data, sequence_length=max_sentence_length, num_classes=num_classes, vocab_size=vocab_size,
          embedding_size=embedding_size, filter_sizes=filter_sizes, num_filters=num_filters, l2_reg_lambda=l2_reg_lambda)


if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    main()

# coding: utf-8

import time
from itertools import chain
from typing import List, Tuple, Dict

import tensorflow as tf
from sklearn.model_selection import train_test_split

from model.text_cnn import TextCNN
import os

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import re


def _read_file(filepath: str) -> pd.DataFrame:
    with open(filepath, 'rb') as f:
        return pd.DataFrame({'review': [str(l) for l in f.readlines()]})


def clean_str(string: str) -> List[str]:
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split(sep=' ')


def batch_iter(data: List[Tuple[np.ndarray, int]], batch_size, num_epochs, do_shuffle=True):
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


def text_to_index(texts: List[str]) -> Tuple[List[np.ndarray], Dict[str, int]]:
    split_texts = list(map(clean_str, texts))
    words = set(list(chain(*split_texts)))
    dictionary = dict(zip(words, range(len(words))))
    corpus = [np.array(list(map(lambda x: dictionary[x], text))) for text in split_texts]
    print(len(corpus))
    return corpus, dictionary


def create_data(max_sentence_length: int = 100):
    print(os.getcwd())
    positive_dataset = _read_file(os.path.join('data', 'rt-polaritydata', 'rt-polarity.pos'))
    positive_dataset['label'] = [np.array([0., 1.]) for _ in range(positive_dataset.shape[0])]
    negative_dataset = _read_file(os.path.join('data', 'rt-polaritydata', 'rt-polarity.neg'))
    negative_dataset['label'] = [np.array([1., 0.]) for _ in range(negative_dataset.shape[0])]

    data = pd.concat((positive_dataset, negative_dataset), axis=0)
    data['review'], dictionary = text_to_index(data['review'].values)
    eos = len(dictionary)
    dictionary['eos'] = eos
    data['review'] = data['review'].apply(lambda x: x[:max_sentence_length])
    data['review'] = data['review'].apply(lambda x: np.pad(x, (0, max_sentence_length - len(x)), 'constant', constant_values=(0, eos)))
    return data, dictionary


def train(batches,
          test_data,
          sequence_length,
          num_classes,
          vocab_size,
          embedding_size,
          filter_sizes,
          num_filters,
          l2_reg_lambda):

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=sequence_length,
                num_classes=num_classes,
                vocab_size=vocab_size,
                filter_sizes=filter_sizes,
                num_filters=num_filters,
                embedding_size=embedding_size,
                l2_reg_lambda=l2_reg_lambda)

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
    batch_size = 100
    num_epochs = 5

    dataset, dictionary = create_data(max_sentence_length=max_sentence_length)
    data = list(zip(dataset['review'], dataset['label']))
    train_data, test_data = train_test_split(data, random_state=123, test_size=0.1)
    train_iter = batch_iter(train_data, batch_size=batch_size, num_epochs=num_epochs, do_shuffle=True)

    num_classes = 2
    vocab_size = len(dictionary)
    embedding_size = 300
    filter_sizes = [3, 4, 5]
    num_filters = 100
    l2_reg_lambda = 10

    train(train_iter, test_data, sequence_length=max_sentence_length, num_classes=num_classes, vocab_size=vocab_size,
          embedding_size=embedding_size, filter_sizes=filter_sizes, num_filters=num_filters, l2_reg_lambda=l2_reg_lambda)


if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    main()

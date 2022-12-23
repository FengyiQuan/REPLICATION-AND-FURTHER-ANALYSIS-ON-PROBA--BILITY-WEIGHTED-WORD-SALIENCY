# import spacy
import os
import re
from config import config
import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from data_reader import read_imdb_file, read_yahoo_file, read_agnews_file, load_all_imdb



def get_tokenizer(dataset):
    texts = None
    if dataset == 'imdb':
        texts, _ = read_imdb_file('train')
    elif dataset == 'yahoo':
        texts, _, _ = read_yahoo_file()
    elif dataset == 'agnews':
        texts, _, _ = read_agnews_file('train')
    tokenizer = Tokenizer(num_words=config.num_words[dataset])
    tokenizer.fit_on_texts(texts)
    return tokenizer


def word_process(train_texts, train_labels, test_texts, test_labels, dataset):
    maxlen = config.word_max_len[dataset]
    tokenizer = get_tokenizer(dataset)

    x_train_seq = tokenizer.texts_to_sequences(train_texts)
    x_test_seq = tokenizer.texts_to_sequences(test_texts)
    x_train = sequence.pad_sequences(x_train_seq, maxlen=maxlen, padding='post', truncating='post')
    x_test = sequence.pad_sequences(x_test_seq, maxlen=maxlen, padding='post', truncating='post')
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    return x_train, y_train, x_test, y_test


def text_to_vector(text, tokenizer, dataset):
    maxlen = config.word_max_len[dataset]
    vector = tokenizer.texts_to_sequences([text])
    vector = sequence.pad_sequences(vector, maxlen=maxlen, padding='post', truncating='post')
    return vector


def text_to_vector_for_all(text_list, tokenizer, dataset):
    maxlen = config.word_max_len[dataset]
    vector = tokenizer.texts_to_sequences(text_list)
    vector = sequence.pad_sequences(vector, maxlen=maxlen, padding='post', truncating='post')
    return vector


# print(get_tokenizer('imdb'))
# train_texts, train_labels, test_texts, test_labels = load_all_imdb()
#
# x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, 'imdb')
#
# x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, 'yahoo')
#
# x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, 'agnews')
# print(x_train)
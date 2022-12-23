import re
from typing import List, Tuple
import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split


def rm_br(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


# file_type is either train or test
def read_imdb_file(file_type: str) -> Tuple[List[str], List[List[int]]]:
    print('Loading %s imdb' % file_type)
    if file_type not in ['train', 'test']:
        raise Exception()
    else:
        all_file_path_list = []
        positive_path = './datasets/aclImdb/' + file_type + '/pos/'
        negative_path = './datasets/aclImdb/' + file_type + '/neg/'
        all_file_path_list.extend([positive_path + file_path for file_path in os.listdir(positive_path)])
        all_file_path_list.extend([negative_path + file_path for file_path in os.listdir(negative_path)])

        texts = []
        labels = [[0, 1] for _ in range(12500)] + [[1, 0] for _ in range(12500)]  # [0,1] is positive, [1,0] is negative

        for file_path in all_file_path_list:
            with open(file_path, )as f:
                texts.append(rm_br(" ".join(f.readlines())))
        print("Loaded %i exs from file imdb" % len(texts))
        return texts, labels


def load_all_imdb() -> Tuple[List[str], List[List[int]], List[str], List[List[int]]]:
    print("Loading imdb")
    train_texts, train_labels = read_imdb_file('train')
    test_texts, test_labels = read_imdb_file('test')
    return train_texts, train_labels, test_texts, test_labels


def read_yahoo_file():
    texts = []
    labels = []
    labels_index = {}
    yahoo_dir = './datasets/yahoo_10'
    for categorical_name in os.listdir(yahoo_dir):
        label_id = len(labels_index)
        labels_index[categorical_name] = label_id
        for file_name in sorted(os.listdir(yahoo_dir + '/' + categorical_name)):
            with open(yahoo_dir + '/' + categorical_name + '/' + file_name) as f:
                texts.append(f.read())
                f.close()
                labels.append(label_id)

    labels = to_categorical(np.asarray(labels))
    print("Loaded %i exs from file yahoo" % len(texts))
    return texts, labels, labels_index


def load_all_yahoo() -> Tuple[List[str], List[List[int]], List[str], List[List[int]]]:
    print("Loading yahoo")
    texts, labels, _ = read_yahoo_file()
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2)
    return train_texts, train_labels, test_texts, test_labels


def to_categorical(y, num_classes=None, dtype="float32"):
    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def read_agnews_file(file_type) -> Tuple[List[str], List[np.ndarray], List[str]]:
    print('Loading %s agnews' % file_type)
    if file_type not in ['train', 'test']:
        raise Exception()
    else:
        path = './datasets/ag_news_csv/' + file_type + '.csv'
        texts = []
        labels_index = []  # The index of label of all input sentences, which takes the values 1,2,3,4
        doc_count = 0  # number of input sentences

        with open(path, newline='') as csv_file:
            reader = csv.reader(csv_file, delimiter=',', quotechar='"')
            for row in reader:
                content = row[1] + ". " + row[2]
                texts.append(content)
                labels_index.append(row[0])
                doc_count += 1

        # Start document processing
        labels = []
        for i in range(doc_count):
            label_class = np.zeros(4, dtype='float32')  # number of classes: 4
            label_class[int(labels_index[i]) - 1] = 1
            labels.append(label_class)
        print("Loaded %i exs from file ag news" % len(texts))
        return texts, labels, labels_index


def load_all_agnews() -> Tuple[List[str], List[np.ndarray], List[str], List[np.ndarray]]:
    print("Loading AG's News dataset")
    train_texts, train_labels, _ = read_agnews_file('train')  # 120000
    test_texts, test_labels, _ = read_agnews_file('test')  # 7600
    return train_texts, train_labels, test_texts, test_labels


def combine_x_y(train_texts, train_labels, test_texts, test_labels):
    if (len(train_texts) != len(train_labels) or len(test_texts) != len(test_labels)):
        raise Exception()
    train_dataset = [[train_texts[i], train_labels[i]] for i in range(len(train_texts))]
    val_dataset = [[test_texts[i], test_labels[i]] for i in range(len(test_texts))]
    return train_dataset, val_dataset
# load_all_yahoo()

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import re
import argparse
from data_reader import split_imdb_files, split_yahoo_files, split_agnews_files
from process import word_process, get_tokenizer, text_to_vector_for_all
from models import word_cnn, char_cnn, bd_lstm, lstm


parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', type=int, default=1000, help='number of samples to generate')
parser.add_argument('--model', type=str, default='word_cnn', help='name of the model')
parser.add_argument('--dataset', type=str, default='imdb', help='name of the dataset')
parser.add_argument('--model_path', type=str,  help='path to load the model weight')


def read_adversarial_file(adversarial_text_path):
    adversarial_text = list(open(adversarial_text_path, "r", encoding='utf-8').readlines())
    # remove sub_rate and NE_rate at the end of the text
    adversarial_text = [re.sub(' sub_rate.*', '', s) for s in adversarial_text]
    return adversarial_text


def get_sub_rate(adversarial_text_path):
    adversarial_text = list(open(adversarial_text_path, "r", encoding='utf-8').readlines())
    all_sub_rate = []
    sub_rate_list = []
    for index, text in enumerate(adversarial_text):
        sub_rate = re.findall('\d+.\d+(?=; NE_rate)', text)
        if len(sub_rate) != 0:
            sub_rate = sub_rate[0]
            all_sub_rate.append(float(sub_rate))
            sub_rate_list.append((index, float(sub_rate)))
    mean_sub_rate = sum(all_sub_rate) / len(all_sub_rate)
    sub_rate_list.sort(key=lambda t: t[1], reverse=True)
    return mean_sub_rate

if __name__ == '__main__':
    args = parser.parse_args()
    n_samples = args.n_samples

    # get tokenizer
    dataset = args.dataset
    tokenizer = get_tokenizer(dataset)

    # Read data set
    x_train = y_train = x_test = y_test = None
    test_texts = None
    first_get_dataset = False
    if dataset == 'imdb':
        train_texts, train_labels, test_texts, test_labels = split_imdb_files()
        x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
    elif dataset == 'agnews':
        train_texts, train_labels, test_texts, test_labels = split_agnews_files()
        x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
    elif dataset == 'yahoo':
        train_texts, train_labels, test_texts, test_labels = split_yahoo_files()
        x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)

    if args.model == "word_cnn":
        model = word_cnn(dataset)
    elif args.model == "bdlstm":
        model = bd_lstm(dataset)
    elif args.model == "lstm":
        model = lstm(dataset)
    model.load_state_dict(torch.load(args.model_path))

    # evaluate classification accuracy of model on adversarial examples
    adv_text_path = r'./fool_result/{}/{}/adv_{}.txt'.format(dataset, args.model, str(n_samples))
    adv_text = read_adversarial_file(adv_text_path)
    x_adv = text_to_vector_for_all(adv_text, tokenizer, dataset)

    sub_rate = get_sub_rate(adv_text_path)
    print('sub. rate:', sub_rate)

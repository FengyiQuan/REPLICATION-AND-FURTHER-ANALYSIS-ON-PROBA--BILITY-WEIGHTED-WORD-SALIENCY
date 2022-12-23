from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import argparse
import os
import numpy as np
from data_reader import split_imdb_files, split_yahoo_files, split_agnews_files
from process import word_process, get_tokenizer
from models import word_cnn, bd_lstm, lstm
from utils import model_evaluate, adversarial_paraphrase


class pwws_attacker(object):
    def __init__(self, params):
        self.p = params
        
    def fool_text_classifier(self):

        # get tokenizer
        dataset = args.dataset
        tokenizer = get_tokenizer(dataset)

        # Read data set
        x_test = y_test = None
        test_texts = None
        if dataset == 'imdb':
            train_texts, train_labels, test_texts, test_labels = split_imdb_files()
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
        elif dataset == 'agnews':
            train_texts, train_labels, test_texts, test_labels = split_agnews_files()
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
        elif dataset == 'yahoo':
            train_texts, train_labels, test_texts, test_labels = split_yahoo_files()
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)

        # Write clean examples into a txt file
        clean_texts_path = r'./fool_result/{}/clean_{}.txt'.format(dataset, str(self.p.n_samples))
        if not os.path.isfile(clean_texts_path):
            write_origin_input_texts(clean_texts_path, test_texts)

        # Select the model and load the trained weights
        if args.model == "word_cnn":
            model = word_cnn(dataset)
        elif args.model == "bdlstm":
            model = bd_lstm(dataset)
        elif args.model == "lstm":
            model = lstm(dataset)
        model.load_state_dict(torch.load(args.model_path))

        # evaluate classification accuracy of model on clean samples
        scores_origin = model.evaluate(x_test[:self.p.n], y_test[:self.p.n])
        print('clean samples origin test_loss: %f, accuracy: %f' % (scores_origin[0], scores_origin[1]))

        grad_guide = model_evaluate(model)
        classes_prediction = grad_guide.predict_classes(x_test[: self.p.n])

        sub_rate_list = []
        NE_rate_list = []

        adv_text_path = r'./fool_result/{}/{}/adv_{}.txt'.format(dataset, args.model, str(self.p.n))
        change_tuple_path = r'./fool_result/{}/{}/change_tuple_{}.txt'.format(dataset, args.model, str(self.p.n))
        file_1 = open(adv_text_path, "a")
        file_2 = open(change_tuple_path, "a")
        for index, text in enumerate(test_texts[: self.p.n]):
            sub_rate = 0
            NE_rate = 0
            if np.argmax(y_test[index]) == classes_prediction[index]:
                adv_doc, adv_y, sub_rate, NE_rate, change_tuple_list = adversarial_paraphrase(input_text=text,
                                                                                            true_y=np.argmax(y_test[index]),
                                                                                            grad_guide=grad_guide,
                                                                                            tokenizer=tokenizer,
                                                                                            dataset=dataset,
                                                                                            level=args.level)
                text = adv_doc
                sub_rate_list.append(sub_rate)
                NE_rate_list.append(NE_rate)
                file_2.write(str(index) + str(change_tuple_list) + '\n')
            file_1.write(text + " sub_rate: " + str(sub_rate))
        mean_sub_rate = sum(sub_rate_list) / len(sub_rate_list)
        print('mean substitution rate:', mean_sub_rate)
        file_1.close()
        file_2.close()
def write_origin_input_texts(origin_input_texts_path, test_texts, test_samples_cap=None):
    if test_samples_cap is None:
        test_samples_cap = len(test_texts)
    with open(origin_input_texts_path, 'a') as f:
        for i in range(test_samples_cap):
            f.write(test_texts[i] + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PWWS attack config')
    parser.add_argument('-n_samples', dest="n_samples", default=1000, type=int, help='number of samples')
    parser.add_argument('-model', dest="model", default="lstm", type=str, help='model type')
    parser.add_argument('-data', dest='data', default='imdb', type=str, help="name of the dataset")
    parser.add_argument('-model_path', dest='model_path',type=str,  help='path to load the model weight')
    args = parser.parse_args()
    pwws_attacker(args).fool_text_classifier()

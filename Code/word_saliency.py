from config import config
import copy
import spacy
from process import text_to_vector

nlp = spacy.load('en_core_web_sm')


def get_word_saliency(doc, grad_guide, tokenizer, input_y, dataset):
    word_saliency_list = []

    max_len = config.word_max_len[dataset]
    text = [doc[position].text for position in range(len(doc))]
    text = ' '.join(text)
    origin_vector = text_to_vector(text, tokenizer, dataset)
    origin_prob = grad_guide.predict_prob(input_vector=origin_vector)
    for position in range(len(doc)):
        if position >= max_len:
            break
        without_word_vector = copy.deepcopy(origin_vector)
        without_word_vector[0][position] = 0
        prob_without_word = grad_guide.predict_prob(input_vector=without_word_vector)

        # calculate priorty score Q
        word_saliency = origin_prob[input_y] - prob_without_word[input_y]
        word_saliency_list.append((position, doc[position], word_saliency, doc[position].tag_))

    position_word_list = []
    for word in word_saliency_list:
        position_word_list.append((word[0], word[1]))

    return position_word_list, word_saliency_list

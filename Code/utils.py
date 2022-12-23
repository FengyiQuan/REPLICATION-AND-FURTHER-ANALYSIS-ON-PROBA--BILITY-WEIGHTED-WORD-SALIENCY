import spacy
import numpy as np
from pwws_rep.pwws import _compile_perturbed_tokens, PWWS
from process import text_to_vector
from word_saliency import evaluate_word_saliency

nlp = spacy.load('en', tagger=False, entity=False)


class model_evaluate:

    def __init__(self, model):
        self.model = model
        
    def predict_prob(self, input_text):
        prob = self.model(input_text).squeeze()
        return prob

    def predict_classes(self, input_text):
        prediction = self.model(input_text)
        classes = np.argmax(prediction, axis=1)
        return classes


def adversarial_paraphrase(input_text, true_y, grad_guide, tokenizer, dataset, level, verbose=True):

    def halt_condition_fn(perturbed_text):
        perturbed_vector = text_to_vector(perturbed_text, tokenizer, dataset)
        adv_y = grad_guide.predict_classes(input_vector=perturbed_vector)
        if adv_y != true_y:
            return True
        else:
            return False

    def heuristic_fn(text, candidate):
        doc = nlp(text)
        origin_vector = text_to_vector(text, tokenizer, dataset)
        perturbed_tokens = _compile_perturbed_tokens(doc, [candidate])
        perturbed_doc = nlp(' '.join(perturbed_tokens))
        perturbed_vector = text_to_vector(perturbed_doc.text, tokenizer, dataset)

        origin_prob = grad_guide.predict_prob(input_vector=origin_vector)
        perturbed_prob = grad_guide.predict_prob(input_vector=perturbed_vector)
        delta_p = origin_prob[true_y] - perturbed_prob[true_y]

        return delta_p

    doc = nlp(input_text)

    _, word_saliency_list = evaluate_word_saliency(doc, grad_guide, tokenizer, true_y, dataset, level)
    perturbed_text, sub_rate, NE_rate, change_tuple_list = PWWS(doc, true_y, dataset, word_saliency_list=word_saliency_list, heuristic_fn=heuristic_fn, halt_condition_fn=halt_condition_fn, verbose=verbose)

    perturbed_vector = text_to_vector(perturbed_text, tokenizer, dataset)
    perturbed_y = grad_guide.predict_classes(input_vector=perturbed_vector)
    return perturbed_text, perturbed_y, sub_rate, NE_rate, change_tuple_list

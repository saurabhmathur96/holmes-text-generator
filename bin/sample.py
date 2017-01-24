from __future__ import print_function
import numpy as np
import random
import sys
import pprint
from utils.config import *
from utils.beamsearch import beamsearch
from utils.greedysearch import greedysearch
from keras.models import load_model


#
# Load model
#

model = load_model(MODEL_FILE)

#
# Load data
# 

text = open(DATA_FILE).read()
chars = open(CHARS_FILE).read().split("\n")
# print ("chars:")
# pprint.pprint(chars)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


#
# Generate sample based on seed.
#
def predict(model, sample):
    x = np.zeros((1, MAXLEN, len(chars)))
    for t, char in enumerate(sample[-MAXLEN:]):
        x[0, t, char] = 1.
    preds = model.predict(x, verbose=0)[0]
    return preds

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
def keras_rnn_predict(samples, rnn_model=model, maxlen=MAXLEN):
    return np.array([predict(rnn_model, sample) for sample in samples])

for _ in range(10):
    start_index = random.randint(0, len(text) - MAXLEN - 1)
    start_index = text.find(".", start_index) + 1 # Choose start of a sentence

    sentence = text[start_index: start_index + MAXLEN].strip()
    print ("-" * 50)
    print ("seed: " + sentence)
    print ("original: " + text[start_index: start_index + MAXLEN*3])
    # generated = greedysearch([char_indices[c] for c in sentence], predict, model, MAXLEN*3)
    # for s in generated:
    #     print ("".join(indices_char[i] for i in s))
    for diversity in [0.2, 0.5, 1.0, 1.2]:

        generated = ''
        sentence = text[start_index: start_index +  MAXLEN]
        generated += sentence
        sys.stdout.write(str(diversity) + ">")
        sys.stdout.write(sentence)

        for i in range(MAXLEN*3):
            x = np.zeros((1, MAXLEN, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print("\n")
    print ("\n\n")

    sentence = text[start_index: start_index +  MAXLEN]
    indices, scores = beamsearch(start=[char_indices[c] for c in sentence], predict=keras_rnn_predict, maxsample=MAXLEN*3, k=2)
    for each, score in zip(indices[:2], scores[:2]):
        print (str(round(score, 3)) + ">" + "".join(indices_char[i] for i in each))
    print ("-" * 50)


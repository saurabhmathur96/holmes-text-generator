from __future__ import print_function
import os
import random
import pprint
import numpy as np
from utils.config import *
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU
from keras.optimizers import RMSprop

#
# Configuration
#

SAMPLES_PER_EPOCH = 500000
BATCH_SIZE=500
N_EPOCH=10


#
# Read data
#

text = open(DATA_FILE).read()
if not os.path.exists(CHARS_FILE):
    chars = sorted(list(set(text)))
    open(CHARS_FILE, "w").write("\n".join(chars))
else:
    chars = open(CHARS_FILE).read().split("\n")
# print ("chars:")
# pprint.pprint(chars)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


#
# Build or load model
#

if not os.path.exists(MODEL_FILE):
    print ("Creating new model.")
    model = Sequential()
    model.add(GRU(256, input_shape=(MAXLEN, len(chars)), return_sequences=True))
    model.add(GRU(256, return_sequences=False))
    model.add(Dense(len(chars)))
    model.add(Activation("softmax"))
else:
    print ("Loading existing model from {0}".format(MODEL_FILE))
    model = load_model(MODEL_FILE)

optimizer = RMSprop(lr=2e-3, rho=.95, clipnorm=5.)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)


#
# Train model
# 

for epoch in range(N_EPOCH):
    sentences = []
    next_chars = []
    for _ in range(SAMPLES_PER_EPOCH):
        i = random.randint(0, len(text) - MAXLEN - 1)
        sentences.append(text[i: i + MAXLEN])
        next_chars.append(text[i + MAXLEN])

    X = np.zeros((len(sentences), MAXLEN, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    model.fit(X, y, batch_size=BATCH_SIZE, nb_epoch=1)

    model.save(MODEL_FILE)
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.callbacks import Callback
from random import randint

# Open Trainging Dataset
with open('sonnets.txt', 'r') as file:
    corpus = file.read()

# Extract unqie chars to list
chars = list(set(corpus))
data_size, vocab_size = len(corpus), len(chars)

# Index dictionaries by character and character indices
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

sentence_length = 50 #  length of words to learn what the next should be
sentences = []
next_chars = []

for i in range(data_size - sentence_length):
    sentences.append(corpus[i: i + sentence_length])
    next_chars.append(corpus[i + sentence_length])

num_sentences = len(sentences)

X = np.zeros((num_sentences, sentence_length, vocab_size), dtype= np.bool)
y = np.zeros((num_sentences, vocab_size), dtype= np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentences):
        X[i, t, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1

# Set up model and feed directly into LSTM
model = Sequential()
model.add(LSTM(256, input_shape=(sentence_length, vocab_size)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X, y, epochs= 30, batch_size=256)

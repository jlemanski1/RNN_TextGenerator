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

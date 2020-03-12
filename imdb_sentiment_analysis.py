import keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.datasets import imdb
from keras.preprocessing import sequence

VOCAB_SIZE = 5000   # Number of words to load
INDEX_FROM = 3      # Start at 3 to account for padding/unknown, & start of sentence
MAX_SEQ_LEN = 128   # Max input length

# Load and assign dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE, index_from=INDEX_FROM)

word_to_idx = imdb.get_word_index()
idx_to_word = {v + INDEX_FROM: k for k,v in word_to_idx.items()}

idx_to_word[0] = '<PAD>'
idx_to_word[1] = '<START>'
idx_to_word[2] = '<UNK>'

X_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQ_LEN)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_SEQ_LEN)

# Print result
print(' '.join([idx_to_word[idx] for idx in X_train[0]]))


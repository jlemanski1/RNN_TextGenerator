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
    for t, char in enumerate(sentence):
        X[i, t, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1

# Set up model and feed directly into LSTM
model = Sequential()
model.add(LSTM(256, input_shape=(sentence_length, vocab_size)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X, y, epochs= 30, batch_size=256)


#
#   Generates and returns sample text from the model
#
def sample_from_model(mode, sample_length= 100):
    seed = randint(0, data_size - sentence_length)
    # Sentence starter seed
    seed_sentence = corpus[seed: seed + sentence_length]

    X_pred = np.zeros((1, sentence_length, vocab_size), dtype= np.bool)
    # Sample by index
    for t, char in enumerate(seed_sentence):
        X_pred[0, t, char_to_idx[char]] = 1
    
    generated_text = ''

    # Use reversed dictionary to form the sentence
    for i in range(sample_length):
        prediction = np.argmax(model.predict(X_pred))

        generated_text += idx_to_char[prediction]

        # Update model to learn from samples made
        activation = np.zeros((1, 1, vocab_size), dtype= np.bool)
        activation[0, 0, prediction] = 1
        X_pred = np.concatenate((X_pred[:, 1:, :], activation), axis= 1)
    
    return generated_text


#
#   Text Generator Callback. Called on end of every epoch
#
class SamplerCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        generated_text = sample_from_model(self.model)
        print('\nGenerated Text')
        print('-' * 32)
        print(generated_text)


# Set paramters and Sample from the model
sampler_callback = SamplerCallback()
model.fit(X, y, epochs= 30, batch_size=256, callbacks= [sampler_callback])

generated_text = sample_from_model(model, sample_length= 1000)
print('\nGenerated Text')
print('-' * 32)
print(generated_text)
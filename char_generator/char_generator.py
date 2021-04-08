from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow import config

physical_devices = config.list_physical_devices('GPU')
config.experimental.set_memory_growth(physical_devices[0], enable=True)

import numpy as np

from os import path
from time import time
from sys import argv

with open('cases.txt', 'r') as f:
    sentences = f.readlines()
    sentences = [sentence.strip() for sentence in sentences]
    np.random.shuffle(sentences)

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
sequences = [tok for seq in sequences for tok in seq]

seq_len = 128
char_sequences = [sequences[i:i+seq_len] for i in range(len(sequences) - seq_len)]

vocab_size = len(tokenizer.word_index) + 1

padded_sequences = pad_sequences(char_sequences)

x = padded_sequences[:,:-1, np.newaxis]

model_path = 'char_keras.h5'

if path.exists(model_path):
    model = load_model(model_path)

    index_word = {v: k for k, v in tokenizer.word_index.items()}
    index_word[0] = '\ufffd'
    new_sentence = []
    in_text = x[np.random.randint(len(x))].T[0]
    print('Seed:', ''.join([index_word[w] for w in in_text]))
    char_count = 80 if len(argv) < 2 else int(argv[1])

    start = time()

    for _ in range(char_count):
        encoded = pad_sequences([in_text], maxlen=x.shape[1])[..., np.newaxis]
        prediction = model.predict(encoded)[0]
        new_word = np.random.choice(len(prediction), p=prediction)
        in_text = np.append(in_text, new_word)
        new_sentence.append(index_word[new_word])
    
    end = time()

    print(f'Generated {char_count} characters in {end - start:.02f} seconds')
    print(''.join(new_sentence))

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow import config

physical_devices = config.list_physical_devices('GPU')
config.experimental.set_memory_growth(physical_devices[0], enable=True)

import numpy as np
import matplotlib.pyplot as plt

from os import path
import time

def plot_history(history, acc_name='keras_accuracy.png', loss_name='keras_loss.png'):
    plt.clf()
    
    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']
    epochs = range(1, len(accuracy) + 1)
    plt.plot(epochs, accuracy, 'tab:blue', label = 'Training Accuracy')
    plt.plot(epochs, val_accuracy, 'tab:orange', label = 'Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(acc_name)

    plt.clf()

    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'tab:blue', label = 'Training Loss')
    plt.plot(epochs, val_loss, 'tab:orange', label = 'Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_name)

def generate_chars(model, word_index, seed, chars=80):
    mlen = len(seed)
    index_word = {v: k for k, v in word_index.items()}
    index_word[0] = '\ufffd'
    new_sentence = []
    seed_str = ''.join([index_word[w] for w in seed])

    for _ in range(chars):
        encoded = pad_sequences([seed], maxlen=mlen)[..., np.newaxis]
        prediction = model.predict(encoded)[0]
        new_word = np.random.choice(len(prediction), p=prediction)
        seed = np.append(seed, new_word)
        new_sentence.append(index_word[new_word])
    
    return seed_str, ''.join(new_sentence)

with open('cases.txt', 'r') as f:
    sentences = f.readlines()
sentences = [sentence.strip() for sentence in sentences]

cases = []
case = []
for sentence in sentences:
    if sentence == '-----':
        cases.append(case)
        case = []
    else:
        case.append(sentence)

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts([s for s in sentences if s != '-----'])
sequences = [tokenizer.texts_to_sequences(case) for case in cases]
sequences = [[tok for seq in case for tok in seq] for case in sequences]
# sequences = [tok for seq in sequences for tok in seq]

seq_len = 250
char_sequences = [sequences[i][j:j+seq_len] for i in range(len(sequences)) for j in range(len(sequences[i]) - seq_len)]
# char_sequences = [sequences[i:i+seq_len] for i in range(len(sequences) - seq_len)]
np.random.shuffle(char_sequences)

vocab_size = len(tokenizer.word_index) + 1

training_proportion = 0.8
training_count = round(training_proportion * len(char_sequences))

# Show distribution of output characters
# from collections import Counter
# index_word = {v: k for k, v in tokenizer.word_index.items()}
# foo = Counter(pad_sequences(char_sequences)[:,-1].T)
# print('\n'.join([f'{index_word[char]}\t{count}' for char, count in foo.most_common()]))

training_sequences, test_sequences = char_sequences[:training_count], char_sequences[training_count:]

padded_training_sequences = pad_sequences(training_sequences)
padded_test_sequences = pad_sequences(test_sequences)

train_x, train_y = padded_training_sequences[:,:-1, np.newaxis], to_categorical(padded_training_sequences[:,-1], num_classes=vocab_size)
test_x, test_y = padded_test_sequences[:,:-1, np.newaxis], to_categorical(padded_test_sequences[:,-1], num_classes=vocab_size)

model = Sequential([
    layers.Embedding(vocab_size, 32, input_length=train_x.shape[1]),
    layers.GRU(256, return_sequences=True),
    layers.Dropout(0.5),
    layers.GRU(256, kernel_regularizer='l2', activity_regularizer='l1'),
    layers.Dropout(0.6),
    layers.Dense(vocab_size, activation='softmax'),
])

model_path = 'char_keras.h5'

if path.exists(model_path):
    model = load_model(model_path)
else:
    model.compile(loss='categorical_crossentropy', optimizer=Adam(clipvalue=0.25), metrics=['accuracy'])

model.summary()

checkpoint_filepath = "char_keras_checkpoint.h5"
checkpoint = callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min')

class GeneratorCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 != 0:
            return
        
        seed, predicted = generate_chars(self.model, tokenizer.word_index, test_x[np.random.randint(len(test_x))].T[0])

        print(f'\nseed: {seed}\npredicted: {predicted}')

generator = GeneratorCallback()

class PlotterCallback(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'lr': []}
    
    def on_epoch_end(self, epoch, logs={}):
        for k, v in logs.items():
            self.history[k].append(v)
        
        if (epoch + 1) % 10 == 0:
            plot_history(self.history, acc_name='plots/char_keras_accuracy.png', loss_name='plots/char_keras_loss.png')

plotter = PlotterCallback()

def schedule(epoch, lr):
    decay = 0.08
    if (epoch + 1) % 8 == 0:
        return lr * (1 - decay)
    return lr

scheduler = callbacks.LearningRateScheduler(schedule)

callbacks_list = [checkpoint, generator, scheduler, plotter]

start = time.time()
history = model.fit(train_x, train_y, epochs=480, batch_size=128, validation_data=(test_x, test_y), callbacks=callbacks_list)
end = time.time()

print('#' * 80)
print(f'Elapsed Training Time: {end - start:.02f}s')
print('#' * 80)

model.load_weights(checkpoint_filepath)
model.save(model_path)
plot_history(history.history)

seed, predicted = generate_chars(model, tokenizer.word_index, test_x[np.random.randint(len(test_x))].T[0], chars=320)
print(f'\nseed: {seed}\npredicted: {predicted}')

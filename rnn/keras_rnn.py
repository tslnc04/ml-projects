from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, Dense, Dropout, Flatten, LSTM, MaxPool1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

def split_sequences(seqs, mlen):
    split = []
    for seq in seqs:
        for i in range(0, len(seq), mlen):
            chunk = seq[i:i+mlen]
            if len(chunk) > 0.75 * mlen:
                split.append(chunk)
    return split

with open('essays.txt', 'r') as f:
    sentences = f.readlines()

# underscore removed for start and end purposes
tokenizer = Tokenizer(oov_token='__unk__')
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
sequences = split_sequences(sequences, 20)
np.random.shuffle(sequences)
vocab_size = len(tokenizer.word_index) + 1

training_proportion = 0.8
training_count = round(training_proportion * len(sequences))

training_sequences, test_sequences = sequences[:training_count], sequences[training_count:]

padded_training_sequences = pad_sequences(training_sequences)
padded_test_sequences = pad_sequences(test_sequences)

train_x, train_y = padded_training_sequences[:,:-1], to_categorical(padded_training_sequences[:,-1], num_classes=vocab_size)
test_x, test_y = padded_test_sequences[:,:-1], to_categorical(padded_test_sequences[:,-1], num_classes=vocab_size)

model = Sequential([
    Embedding(vocab_size, 64, input_length=train_x.shape[1]),
    Conv1D(128, 8),
    MaxPool1D(),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(vocab_size, activation='softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]

history = model.fit(train_x, train_y, epochs=250, batch_size=5, validation_data=(test_x, test_y))

import matplotlib.pyplot as plt

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, 'y', label = 'Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('keras_accuracy.png')

plt.clf()

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label = 'Training Loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('keras_loss.png')

# filename = "weights-improvement-12-3.8726.hdf5"
# model.load_weights(filename)
# model.compile(loss='categorical_crossentropy', optimizer='adam')

index_word = {v: k for k, v in tokenizer.word_index.items()}
new_sentence = []
in_text = test_x[np.random.randint(len(test_x))]

for _ in range(40):
    encoded = pad_sequences([in_text], maxlen=train_x.shape[1])
    prediction = model.predict(encoded)[0]
    if sum(prediction) > 1:
        prediction -= 1e-5
        prediction = np.clip(prediction, 0, 1)
    new_word = np.argmax(np.random.multinomial(1, prediction))
    in_text = np.append(in_text, new_word)
    new_sentence.append(index_word[new_word])

print(f'Vocab Size: {vocab_size}, Training Sequences: {training_count}')
print(' '.join(new_sentence))

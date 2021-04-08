from tensorflow.keras.utils import to_categorical
import numpy as np
import time
# from scipy.special import softmax

def softmax(z):
    z = np.clip(z, -30, 30)
    return np.exp(z) / np.sum(np.exp(z), axis=0)

def cross_loss(o_t, y_t):
    return -(y_t @ np.log(o_t + 1e-7))

def cross_loss_prime(o_t, y_t):
    return -(y_t @ (1 / o_t))

def cross_loss_tt(yhat, y):
    L = 0
    for t in range(len(yhat)):
        L += cross_loss(yhat[t], y[t])
    return L / len(yhat)

class RNN():
    def __init__(self, word_dim, hidden_dim=128, bptt_trunc=5):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_trunc = bptt_trunc

        self.U = np.random.uniform(-0.1, 0.1, (hidden_dim, word_dim))
        self.V = np.random.uniform(-0.1, 0.1, (word_dim, hidden_dim))
        self.W = np.random.uniform(-0.1, 0.1, (hidden_dim, hidden_dim))
    
    def forward(self, x):
        T = len(x)

        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)

        o = np.zeros((T, self.word_dim))

        for t in range(T):
            s[t] = np.tanh(self.U @ x[t] + self.W @ s[t-1])
            o[t] = softmax(self.V @ s[t])
        
        return o, s
    
    def backward(self, x, o, y, s):
        T = len(y)

        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)

        delta_o = o - y

        for t in range(T - 1, -1, -1):
            dLdV += np.outer(delta_o[t], s[t])
            delta_t = self.V.T @ delta_o[t] * (1 - s[t] ** 2)

            for step in range(t, max(0, t - self.bptt_trunc) - 1, -1):
                dLdW += np.outer(delta_t, s[step - 1])
                dLdU += np.outer(delta_t, x[step])
                delta_t = self.W.T @ delta_t * (1 - s[step - 1] ** 2)
        
        return dLdU, dLdV, dLdW
    
    def report_loss(self, x, y):
        L = 0
        for i in range(len(x)):
            o, _ = self.forward(x[i])
            L += cross_loss_tt(o, y[i])
        return L / len(x)
    
    def sgd_step(self, x, y, learning_rate):
        o, s = self.forward(x)
        dLdU, dLdV, dLdW = self.backward(x, o, y, s)

        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW
        
        return cross_loss_tt(o, y)
    
    def train(self, x_train, y_train, learning_rate=0.05, epochs=100, validation_data=None):
        history = {'train_loss': [], 'val_loss': [], 'duration': []}
        for epoch in range(epochs):
            train_loss = 0
            start = time.time()
            for i in range(len(y_train)):
                train_loss += self.sgd_step(x_train[i], y_train[i], learning_rate)
            end = time.time()

            duration = end - start
            history['duration'].append(duration)

            if validation_data:
                val_loss = self.report_loss(*validation_data)
                history['val_loss'].append(val_loss)
                print(f'epoch: {epoch+1:03}/{epochs}\t{duration:.02f}s\ttrain_loss: {train_loss / len(y_train):.04f}\tval_loss: {val_loss:.04f}')
            else:
                print(f'epoch: {epoch+1:03}/{epochs}\t{duration:.02f}s\ttrain_loss: {train_loss / len(y_train):.04f}')

            history['train_loss'].append(train_loss / len(y_train))
        return history
    
    def generate(self, word_index):
        vocab_size = len(word_index)
        new_sentence = np.array([word_index['__start__']])
        while new_sentence[-1] != word_index['__end__'] and len(new_sentence) < 100:
            word_probs, _ = self.forward(to_categorical(new_sentence, num_classes=vocab_size))
            sampled = word_index['__unk__']
            while sampled == word_index['__unk__']:
                samples = np.random.multinomial(1, word_probs[-1])
                sampled = np.argmax(samples)
            new_sentence = np.append(new_sentence, sampled)
        index_word = {v: k for k, v in word_index.items()}
        return ' '.join([index_word[new_word] for new_word in new_sentence if index_word[new_word] != '__start__'])

if __name__ == '__main__':
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    with open('essays.txt', 'r') as f:
        sentences = f.readlines()
    
    sentences = ['__start__ ' + sentence.strip('\n') + ' __end__' for sentence in sentences if sentence and sentence != '\n']
    training_proportion = 0.85
    training_count = round(training_proportion * len(sentences))
    training_data, test_data = sentences[:training_count], sentences[training_count:]

    # underscore removed for start and end purposes
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n', oov_token='__unk__')
    tokenizer.fit_on_texts(sentences)

    vocab_size = len(tokenizer.word_index) + 1

    training_sequences = tokenizer.texts_to_sequences(training_data)
    test_sequences = tokenizer.texts_to_sequences(test_data)

    padded_training_sequences = pad_sequences(training_sequences, truncating='pre')
    padded_test_sequences = pad_sequences(test_sequences, truncating='pre')

    onehot_training_sequences = to_categorical(padded_training_sequences, num_classes=vocab_size)
    onehot_test_sequences = to_categorical(padded_test_sequences, num_classes=vocab_size)

    train_x = onehot_training_sequences[:,:,:-1]
    train_y = onehot_training_sequences[:,:,1:]

    test_x = onehot_test_sequences[:,:,:-1]
    test_y = onehot_test_sequences[:,:,1:]

    rnn = RNN(vocab_size - 1, hidden_dim=192)
    history = rnn.train(train_x, train_y, validation_data=(test_x, test_y), epochs=125)
    print(rnn.generate(tokenizer.word_index))

    import matplotlib.pyplot as plt

    train_loss, val_loss = history['train_loss'], history['val_loss']
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')

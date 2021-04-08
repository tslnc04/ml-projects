import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import models
from tensorflow.keras import regularizers

breast_cancer = np.loadtxt('breast-cancer.csv', skiprows=1, delimiter=',')
np.random.shuffle(breast_cancer)

training_prop = 0.75
training_num = round(training_prop * len(breast_cancer))

training_inputs = breast_cancer[:training_num,1:-1]
training_outputs = breast_cancer[:training_num,-1]
# print(inputs, outputs)

test_inputs = breast_cancer[training_num:,1:-1]
test_outputs = breast_cancer[training_num:,-1]

model = models.Sequential([
  layers.InputLayer(input_shape=training_inputs.shape[1]),
  layers.Dense(32, activation='relu'),
  layers.Dense(64, activation='relu'),
  layers.Dropout(0.2),
  layers.Dense(32, activation='relu'),
  layers.Dropout(0.7),
  layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
  layers.Dense(1, activation='sigmoid'),
])

model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(learning_rate=0.02), metrics=['accuracy', metrics.TruePositives(), metrics.TrueNegatives()])

model.summary()

epochs = 50
batch_size = 2

hist = model.fit(
    training_inputs,
    training_outputs,
    epochs=epochs,
    batch_size=batch_size,
    validation_data = (test_inputs, test_outputs),
)

import matplotlib.pyplot as plt

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']

epochs = range(1, len(accuracy) + 1)

plt.plot(epochs, accuracy, 'y', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')

plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.legend()
plt.savefig('cancer_accuracy.png')

plt.clf()

loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'y', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.legend()
plt.savefig('cancer_loss.png')

plt.clf()

true_negatives = hist.history['true_negatives']
val_true_negatives = hist.history['val_true_negatives']

true_positives = hist.history['true_positives']
val_true_positives = hist.history['val_true_positives']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, true_negatives, 'tab:red', label='Training True Negatives')
plt.plot(epochs, val_true_negatives, 'tab:orange', label='Validation True Negatives')

plt.plot(epochs, true_positives, 'tab:blue', label='Training True Positives')
plt.plot(epochs, val_true_positives, 'tab:cyan', label='Validation True Positives')

plt.title('Training and Validation True Negatives and Positives')
plt.xlabel('Epochs')
plt.ylabel('Count')

plt.legend()
plt.savefig('cancer_true.png')

from keras.datasets import mnist
import tensorflow as tf
# import keras
from keras import layers
from keras import models
from keras.utils.np_utils import to_categorical
import numpy as np
import gc
import matplotlib.pyplot as plt

(training_data, training_labels), (testing_data, testing_labels) = mnist.load_data()

# Prepare data

training_data = training_data.reshape(training_data.shape[0], training_data.shape[1] * training_data.shape[2])
training_data = training_data.astype('float32') / 255

testing_data = testing_data.reshape(testing_data.shape[0], testing_data.shape[1] * testing_data.shape[2])
testing_data = testing_data.astype('float32') / 255

training_labels = to_categorical(training_labels)
testing_labels = to_categorical(testing_labels)

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(training_data, training_labels, batch_size=100, epochs=5)
accuracy = history.history['accuracy']
plt.plot(range(1, len(accuracy) + 1), accuracy)
plt.title("Accuracy on training data")
plt.show()
print("Evaluating results:")
eval_results = model.evaluate(testing_data, testing_labels)
print("The mse on the test data is: " + str(eval_results[0]))
print("The accuracy on the test data is: " + str(eval_results[1]))

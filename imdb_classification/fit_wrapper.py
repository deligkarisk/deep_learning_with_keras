from keras.datasets import imdb
from keras import models
from keras import layers
from keras import losses
from keras import metrics
from keras import backend
from tensorflow import optimizers
import gc
from numpy import ndarray
import matplotlib.pyplot as plt


def fit_wrapper(layer_size: int, epochs: int, x_train: ndarray, y_train: ndarray, x_test: ndarray, y_test: ndarray, show_plots: bool) -> (int, int):

    gc.collect()
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    model = models.Sequential()
    model.add(layers.Dense(layer_size, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(layer_size, activation='relu', input_shape=(layer_size,)))
    model.add(layers.Dense(1, activation='sigmoid', input_shape=(layer_size,)))
    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001), loss=losses.binary_crossentropy,
                  metrics=[metrics.binary_accuracy])
    history = model.fit(partial_x_train, partial_y_train, epochs=epochs, batch_size=512, validation_data=(x_val, y_val))
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    epochs_labels = range(1, len(acc) + 1)

    if show_plots:
        plt.plot(epochs_labels, loss_values, 'bo', label='Training loss')
        plt.plot(epochs_labels, val_loss_values, 'b', label='Validation loss')
        plt.title('Training and validation loss (epochs = ' + str(epochs) + ', layer size = ' + str(layer_size) + ')')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.clf()
        plt.plot(epochs_labels, acc, 'bo', label='Training accuracy')
        plt.plot(epochs_labels, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy (epochs = ' + str(epochs) + ', layer size = ' + str(layer_size) + ')')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    evaluation_results = model.evaluate(x_test, y_test)
    backend.clear_session()

    return evaluation_results



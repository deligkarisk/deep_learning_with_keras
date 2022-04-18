from keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt
from fit_wrapper import fit_wrapper

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join(reverse_word_index.get(i - 3, '?') for i in train_data[0])


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

results = []
losses = []
accuracy = []
layer_size_range = range(1, 17)
for i in layer_size_range:
    temp_results = fit_wrapper(i, 4, x_train, y_train, x_test, y_test, False)
    results.append(temp_results)
    losses.append(temp_results[0])
    accuracy.append(temp_results[1])
    print('Finished analysis with layer size: ' + str(i))


print(results)

plt.plot(layer_size_range, losses, 'bo', label='Losses')
plt.plot(layer_size_range, accuracy, 'b', label='Accuracy')
plt.title('Losses and accuracy for increasing layer size')
plt.xlabel('Layer size')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()



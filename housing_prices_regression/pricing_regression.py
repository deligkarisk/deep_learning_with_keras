import numpy
from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from util import smooth_curve

def model_build():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model



(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)
print(train_targets[0:10])

# Feature-wise normalization
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
print(mean)
print(std)
train_data -= mean
train_data /= std

test_data -= mean
test_data /= std
print(train_data.mean(axis=0))
print(train_data.std(axis=0))

k = 4 # number for k-fold validation
num_val_samples = len(train_data) // k
num_epochs = 7
all_scores = []
all_mae_histories = []
average_mae_history = numpy.zeros((k, num_epochs))# epoch-average of the mae

for i in range(k):
    print("Now processing fold #", i)
    validation_data_ind_start = i*num_val_samples
    validation_data_ind_end = (i+1)*num_val_samples
    kfold_training_data = np.concatenate([train_data[:validation_data_ind_start], train_data[validation_data_ind_end:]], axis=0)
    kfold_validation_data = train_data[validation_data_ind_start:validation_data_ind_end]

    kfold_training_targets = np.concatenate([train_targets[:validation_data_ind_start], train_targets[validation_data_ind_end:]], axis=0)
    kfold_validation_targets =  train_targets[validation_data_ind_start:validation_data_ind_end]

    model = model_build()
    history = model.fit(kfold_training_data, kfold_training_targets, epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(kfold_validation_data, kfold_validation_targets, verbose=0)
    all_scores.append(val_mae)
    mae_history = history.history['mae']
    all_mae_histories.append(mae_history)
    average_mae_history[i,:] = np.array(mae_history, ndmin=2)

average_mae_history = np.mean(average_mae_history, axis=0)
print(all_scores)
print(np.mean(all_scores))
plt.plot(range(1, len(average_mae_history) + 1 ), average_mae_history)
plt.title("manual mae plot")
plt.show()

average_mae_history_ref = [
np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
plt.plot(range(1, len(average_mae_history_ref) + 1 ), average_mae_history_ref)
plt.title("reference mae plot")
plt.show()

smooth_mae_history = smooth_curve(average_mae_history[1:])

plt.plot(range(1, len(smooth_mae_history[0]) + 1), smooth_mae_history[0])
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
print("OK")
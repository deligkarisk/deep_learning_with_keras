import keras.callbacks
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras import layers

training_dataset_size = range(10000, 70000, 10000)
(training_data, training_labels), (testing_data, testing_labels) = mnist.load_data()

training_data = training_data.reshape(training_data.shape[0], training_data.shape[1] * training_data.shape[2])
training_data = training_data.astype('float32') / 255

testing_data = testing_data.reshape(testing_data.shape[0], testing_data.shape[1] * testing_data.shape[2])
testing_data = testing_data.astype('float32') / 255

training_labels = to_categorical(training_labels)
testing_labels = to_categorical(testing_labels)

def get_mnist_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    return model





callbacks_list = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                  patience=2),
    keras.callbacks.ModelCheckpoint(filepath="checkpoint_path.keras",
                                    monitor="val_loss",
                                    save_best_only=True)]

model = get_mnist_model()
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.fit(training_data, training_labels, epochs=20, callbacks=callbacks_list, validation_data=(testing_data, testing_labels))

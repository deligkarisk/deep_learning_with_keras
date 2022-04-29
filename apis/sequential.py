from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(3,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
], name="my_model")
model.summary()


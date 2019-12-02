from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt


data = mnist.load_data()
X_train, y_train = data[0][0][10000:], data[0][1][10000:]
X_valid, y_valid = data[0][0][:10000], data[0][1][:10000]
X_test, y_test = data[1]

m, w, h = X_train.shape
X_train = X_train.reshape((m, w * h))
X_train = X_train.astype(np.float32) / 255
y_train = to_categorical(y_train)

m, w, h = X_valid.shape
X_valid = X_valid.reshape((m, w * h))
X_valid = X_valid.astype(np.float32) / 255
y_valid = to_categorical(y_valid)

m, w, h = X_test.shape
X_test = X_test.reshape((m, w * h))
X_test = X_test.astype(np.float32) / 255
y_test = to_categorical(y_test)

model = models.Sequential()
model.add(layers.Dense(units=128, activation="relu", input_shape=X_train.shape[1:]))
model.add(layers.Dense(units=y_train.shape[1], activation="softmax"))
model.summary()

model.compile(
    optimizer="RMSProp",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=50, batch_size=128)
print(history.history)

result = model.evaluate(X_test, y_test)
print(result)

epochs = range(1, len(history.history["loss"]) + 1)

plt.figure()
plt.plot(epochs, history.history["loss"], "b", label="Training loss")
plt.plot(epochs, history.history["accuracy"], "b", label="Training acc")
plt.plot(epochs, history.history["val_loss"], "r", label="Validation loss")
plt.plot(epochs, history.history["val_accuracy"], "r", label="Validation acc")
plt.show()

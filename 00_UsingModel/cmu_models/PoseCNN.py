import os
import cv2
import numpy as np
import keras


if __name__ == "__main__":
    _src = "../../../datasets/pcms/features/"
    if not os.path.exists(_src):
        raise Exception("source directory not found.")

    _label_names = []
    _labels = []
    _features = []
    for _f1 in os.listdir(_src):
        _src_f1 = os.path.join(_src, _f1)
        if _f1 not in _label_names:
            _label_names.append(_f1)
        _label = _label_names.index(_f1)

        for _f2 in os.listdir(_src_f1):
            _image = cv2.imread(os.path.join(_src_f1, _f2), cv2.IMREAD_COLOR)
            _image = cv2.resize(_image, (100, 100))
            _features.append(_image)
            _labels.append(_label)
    _features = np.array(_features)
    _labels = np.array(_labels)

    _input = keras.layers.Input(shape=(100, 100, 3))
    x = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation="relu")(_input)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = keras.layers.Flatten()(x)
    _output = keras.layers.Dense(units=len(_label_names), activation="softmax")(x)
    _model = keras.Model(_input, _output)
    _model.summary()
    _model.compile(
        optimizer="adam",
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=["accuracy"]
    )

    print(_features.shape)
    _model.fit(_features, _labels, epochs=50, batch_size=128)
    _model.save("model.h5")
    print(_label_names)

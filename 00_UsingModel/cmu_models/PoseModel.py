import keras
import numpy as np


f = open("dataset.csv", "r")
lines = f.readlines()
f.close()

features = []
labels = []
label_names = []
for line in lines:
    parts = line.split(",")

    feature = []
    for i in range(len(parts) // 2):
        ix = i * 2
        iy = i * 2 + 1
        if i in [1, 8, 9, 10, 11, 12, 13]:
            feature.append(int(parts[ix]))
            feature.append(int(parts[iy]))
    if feature[0] == -1 or feature[1] == -1:
        continue
    # feature = [int(x) for x in parts[:-9]]
    # if feature[2] == -1 or feature[3] == -1:
    #     continue

    features.append(feature)

    label_name = parts[-1].strip()
    if label_name not in label_names:
        label_names.append(label_name)
    label_index = label_names.index(label_name)
    labels.append(label_index)

features = np.array(features)
labels = np.array(labels)
print(features.shape)
for i in range(features.shape[0]):
    cx = features[i, 0]
    cy = features[i, 1]
    for j in range(features.shape[1]):
        if j % 2 == 0:
            features[i, j] -= cx
        else:
            features[i, j] -= cy
        if features[i, j] < 0:
            features[i, j] = 0

input_1 = keras.layers.Input((14,))
x = keras.layers.Dense(units=64, activation="relu")(input_1)
x = keras.layers.Dense(units=len(label_names), activation="softmax")(x)

model = keras.Model(input_1, x)
model.summary()
model.compile(
    optimizer="Adam",
    loss=keras.losses.sparse_categorical_crossentropy,
    metrics=["accuracy"]
)

model.fit(features, labels, epochs=15000, batch_size=128)
model.save("model.h5")
print(label_names)

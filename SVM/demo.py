import numpy as np


f = open("../../datasets/pcms/pose_dataset/data.csv", "r")
lines = f.readlines()
f.close()

X = []
y = []
for line in lines:
    parts = line.split(" ")
    name = parts[0]
    raw = parts[1:37]
    label = parts[37].strip()

    data = []
    for i in range(18):
        temp_x = int(raw[i * 2])
        temp_y = int(raw[i * 2 + 1])
        # points.append((x, y))
        data.append(temp_x)
        data.append(temp_y)
    X.append(data)
    y.append(label)

c = list(zip(X, y))
np.random.shuffle(c)
X, y = zip(*c)

X = np.array(X)
y = np.array(y)

for i in range(y.shape[0]):
    if y[i] == "stand":
        y[i] = 0
    elif y[i] == "sit":
        y[i] = 1
    elif y[i] == "squat":
        y[i] = 2
    else:
        y[i] = 3
print("X shape:", X.shape)
print("y shape:", y.shape)

s = int(y.shape[0] * 0.8)

# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# model.fit(X[:s], y[:s])

from sklearn.svm import SVC
model = SVC(kernel='rbf')
model.fit(X[:s], y[:s])

# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier()
# model.fit(X[:s], y[:s])


import pickle
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

model = pickle.load(open(filename, "rb"))
y_pred = model.predict(X[s:])

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y[s:], y_pred))
print(classification_report(y[s:], y_pred))

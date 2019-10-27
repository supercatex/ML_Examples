import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 導入數據
data = pd.read_excel("data.xlsx")
y = pd.DataFrame(data, columns=["價格"]).values
X = pd.DataFrame(data, columns=["呎數"]).values
m = y.shape[0]
X = np.append(np.ones((m, 1)), X, axis=1)
n = X.shape[1]

# 正規化數據
X = X / np.max(X, axis=0)
y = y / np.max(y, axis=0)

# 顯示圖表
plt.scatter(X[:, 1], y, 20, "b", marker=".")
plt.show()

# Linear Regression
alpha = 0.5
epoch = 1000
weights = np.zeros((n, 1))
history = []
for i in range(epoch):
    predict = np.dot(X, weights)
    J = np.sum((predict - y) ** 2) / m / 2
    dJ = np.sum((predict - y) * X, axis=0).reshape((n, 1)) / m
    weights = weights - dJ * alpha
    history.append(J)
print("Weights:\n", weights)
print("Cost:\n", J)
print("dJ:\n", dJ)

# 顯示Loss圖表
plt.plot(history)
plt.show()

# 顯示結果
m = 1000
plt.scatter(X[:, 1], y, 20, "b", marker=".")
new_x = np.linspace(np.min(X[:, 1:]), np.max(X[:, 1:]), m)
new_x = np.append(np.ones((m, 1)), new_x.reshape((m, 1)), axis=1)
new_y = np.dot(new_x, weights)
plt.scatter(new_x[:, 1], new_y, 1, "r", marker=".")
plt.show()

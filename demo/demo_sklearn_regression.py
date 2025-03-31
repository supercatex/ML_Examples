from math import radians
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

df = pd.read_csv("linear_regression_dataset_sample.csv")
X = df.iloc[:, 1].values.reshape(-1, 1)
y = df.iloc[:, 2].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

models = {
    "    Linear Regession     ": LinearRegression(),
    "Polynomial Regression (2)": make_pipeline(PolynomialFeatures(2), LinearRegression()),
    "Polynomial Regression (3)": make_pipeline(PolynomialFeatures(3), LinearRegression()),
    "Polynomial Regression (4)": make_pipeline(PolynomialFeatures(4), LinearRegression()),
    "Polynomial Regression (5)": make_pipeline(PolynomialFeatures(5), LinearRegression()),
    "Polynomial Regression (6)": make_pipeline(PolynomialFeatures(6), LinearRegression()),
    "Polynomial Regression (7)": make_pipeline(PolynomialFeatures(7), LinearRegression()),
    "Polynomial Regression (8)": make_pipeline(PolynomialFeatures(8), LinearRegression()),
    "Polynomial Regression (9)": make_pipeline(PolynomialFeatures(9), LinearRegression()),
    "Polynomial Regression(10)": make_pipeline(PolynomialFeatures(10), LinearRegression()),
}

scores = []
width = 5
plt.figure(figsize=(width * (len(models) + 1) // 2, width * 2))
i = 0
for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores.append(score)
    print(name, "score:", score)

    i = i + 1
    ax = plt.subplot(2, (len(models) + 1) // 2, i)
    ax.scatter(X_train, y_train, color="r")
    ax.scatter(X_test, y_test, color="y")
    ax.set_title(name)

    data = list(np.concatenate((X_train, y_train), axis=1))
    data.sort(key=lambda x: x[0])
    data = np.array(data)
    ax.plot(data[:, 0], model.predict(data[:, 0].reshape(-1, 1)), color="b")

plt.figure()
plt.plot(range(1, len(models) + 1), scores)

plt.show()

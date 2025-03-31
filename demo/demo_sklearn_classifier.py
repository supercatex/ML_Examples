import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay

datasets = [
    make_moons(noise=0.3),
    make_circles(noise=0.2, factor=0.5),
    make_classification(n_features=2, n_redundant=0)
]

models = {
    "Logistic Regression": LogisticRegression(),
    "K Nearest Neighbor": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
}

colors = ListedColormap(["#FF0000", "#0000FF"])
width = 2.5 # inches
figure = plt.figure(figsize=(width * (len(models) + 1), width * len(datasets)))
i = 0
for _, data in enumerate(datasets):
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    i = i + 1
    ax = plt.subplot(len(datasets), len(models) + 1, i)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=colors, edgecolors="k")
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=colors, edgecolors="k", alpha=0.6)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())

    for name, model in models.items():
        clf = model.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        # pred = clf.predict(X_test)

        i = i + 1
        ax = plt.subplot(len(datasets), len(models) + 1, i)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=colors, edgecolors="k")
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=colors, edgecolors="k", alpha=0.6)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())

        DecisionBoundaryDisplay.from_estimator(clf, X, cmap=plt.cm.RdBu, alpha=0.8, ax=ax, eps=0.5)
        ax.text(x_max, y_min, "score: %.2f" % score, size=10, horizontalalignment="right")
        if i <= len(models) + 1: ax.set_title(name)

plt.show()

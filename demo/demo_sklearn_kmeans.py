import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

data = load_iris()
df = pd.DataFrame(data["data"], columns=data["feature_names"])
x = df.iloc[:, 0:4]

inertia = []
n_classes = 10
for k in range(1, n_classes + 1, 1):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, n_classes + 1, 1), inertia, marker='o')
plt.show()


width = 5
plt.figure(figsize=(width * (n_classes + 1) // 2, width * 2))
i = 0
for k in range(1, n_classes + 1, 1):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x)

    i = i + 1
    ax = plt.subplot(2, (n_classes + 1) // 2, i)
    ax.scatter(x.iloc[:, 0], x.iloc[:, 1], c=kmeans.labels_)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.array([
    [2, 3], [3, 3], [2, 4],
    [6, 8], [8, 8], [7, 9],
    [8, 2], [9, 2], [9, 3]
])

k = 3

model = KMeans(
    n_clusters=k,
    n_init=10,
    random_state=42
)

model.fit(X)

labels = model.labels_
centers = model.cluster_centers_

print("Labels:", labels)
print("Centers:")
print(centers)

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap='Set2',
    s=100
)


plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c='red',
    marker='X',
    s=250,
    label='Centers'
)

plt.title("K-Means Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

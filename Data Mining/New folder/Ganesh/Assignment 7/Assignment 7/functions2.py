import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

dataset = load_digits()
print(dataset)
digit_data = scale(dataset.data)
num_digits = len(np.unique(dataset.target))

red_data = PCA(n_components=2).fit_transform(digit_data)

h = 0.02
xmin, xmax = red_data[:, 0].min() - 1, red_data[:, 0].max() + 1
ymin, ymax = red_data[:, 1].min() - 1, red_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))

models = [
    (
        KMedoids(metric="manhattan", n_clusters=num_digits, 
        init="heuristic", max_iter=2),"Manhattan metric",
    ),
    (
        KMedoids(metric="euclidean", n_clusters=num_digits,  
        init="heuristic", max_iter=2),"Euclidean metric",
    ),
    (KMedoids(metric="cosine", n_clusters=num_digits, init="heuristic", 
    max_iter=2), "Cosine metric", ),
]

num_rows = int(np.ceil(len(models) / 2.0))
num_cols = 2

plt.clf()
plt.figure(figsize=(15,10))
for i, (model, description) in enumerate(models):
    model.fit(red_data)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.subplot(num_cols, num_rows, i + 1)
    plt.imshow(
        Z,    
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,  
        aspect="auto", 
        origin="lower",  
    )
    plt.plot(
        red_data[:, 0], red_data[:, 1], "k.", markersize=2, alpha=0.3
    )
    centroids = model.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,  
        linewidths=3, 
        color="w", 
        zorder=10, 
    )
    plt.title(description)  
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xticks(())   
    plt.yticks(())
plt.suptitle(
    "K-Medoids algorithm implemented with different metrics\n\n",
    fontsize=20,  
)
plt.show() 

"""SILHOUETTE PLOTS"""

"""
Another intrinsic metric to evaluate the quality of a clustering is silhouette
analysis, which can also be applied to clustering algorithms other than k-means.
Silhouette analysis can be used as a graphical tool to plot a measure of how tightly
grouped the examples in the clusters are. To calculate the silhouette coefficient
of a single example in our dataset, we can apply the following three steps:
1.  Calculate the cluster cohesion, a(i), as the average distance between an example,
    x(i), and all other points in the same cluster.
2.  Calculate the cluster separation, b(i), from the next closest cluster as the
    average distance between the example, x(i), and all examples in the nearest
    cluster.
3.  Calculate the silhouette, s(i), as the difference between cluster cohesion and
    separation divided by the greater of the two.

The silhouette coefficient is bounded in the range –1 to 1. We can see that the
silhouette coefficient is 0 if the cluster separation and cohesion are equal
(b(i) = a(i)). Furthermore, we get close to an ideal silhouette coefficient of 1
if b(i) >> a(i), since b(i) quantifies how dissimilar an example is from other
clusters, and a(i) tells us how similar it is to the other examples in its own
cluster.

The silhouette coefficient is available as silhouette_samples from scikit-learn’s
metric module, and optionally, the silhouette_scores function can be imported for
convenience. The silhouette_scores function calculates the average silhouette
coefficient across all examples, which is equivalent to numpy.mean(silhouette_samples(...)).
By executing the following code, we will now create a plot of the silhouette coefficients
for a k-means clustering with k = 3 :
"""

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

"""Good Clustering"""

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)
from sklearn.metrics import silhouette_samples
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(
    X, y_km, metric='euclidean'
)
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper),
             c_silhouette_vals,
             height=1.0,
             edgecolor='none',
             color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg,
            color="red",
            linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.tight_layout()
plt.show()
"""
Through a visual inspection of the silhouette plot, we can quickly scrutinize the
sizes of the different clusters and identify clusters that contain outliers.
We can see the silhouette coefficients are not close to 0 and are approximately
equally far away from the average silhouette score, which is, in this case, an
indicator of good clustering. Furthermore, to summarize the goodness of our clustering,
we added the average silhouette coefficient to the plot (dotted line).
"""

"""Bad Clustering"""

km = KMeans(n_clusters=2,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)
plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50, c='lightgreen',
            edgecolor='black',
            marker='s',
            label='Cluster 1')
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50,
            c='orange',
            edgecolor='black',
            marker='o',
            label='Cluster 2')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250,
            marker='*',
            c='red',
            label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(
    X, y_km, metric='euclidean'
)
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper),
             c_silhouette_vals,
             height=1.0,
             edgecolor='none',
             color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.tight_layout()
plt.show()
"""
the silhouettes now have visibly different lengths and widths, which is evidence
of a relatively bad or at least suboptimal clustering.
"""

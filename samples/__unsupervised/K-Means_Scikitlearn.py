"""
Clustering (or cluster analysis) is a technique that allows us to find groups of
similar objects that are more related to each other than to objects in other groups.
Examples of business-oriented applications of clustering include the grouping of
documents, music, and movies by different topics, or finding customers that share
similar interests based on common purchase behaviors as a basis for recommendation
engines.
"""

import matplotlib.pyplot as plt
import numpy as np

"""PROTOTYPE-BASED CLUSTERING"""
"""
Prototype-based clustering means that each cluster is represented by a prototype,
which is usually either the centroid (average) of similar points with continuous
features, or the medoid (the most representative or the point that minimizes the
distance to all other points that belong to a particular cluster) in the case of
categorical features.
"""

"""DATASET SCIKITLEARN"""
"""
Although k-means clustering can be applied to data in higher dimensions, we will
walk through the following examples using a simple two-dimensional dataset for the
purpose of visualization.
"""
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)
plt.scatter(X[:, 0],
            X[:, 1],
            c='white',
            marker='o',
            edgecolor='black',
            s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid()
plt.tight_layout()
plt.show()
"""
The dataset that we just created consists of 150 randomly generated points that
are roughly grouped into three regions with higher density, which is visualized
via a two-dimensional scatterplot.
"""

"""K-MEANS CLUSTERING SCIKITLEARN"""
"""
While k-means is very good at identifying clusters with a spherical shape, one of
the drawbacks of this clustering algorithm is that we have to specify the number
of clusters, k, a priori. An inappropriate choice for k can result in poor clustering
performance.
we will discuss the elbow method and silhouette plots, which are useful techniques
to evaluate the quality of a clustering to help us determine the optimal number
of clusters, k.


1.  Randomly pick k centroids from the examples as initial cluster centers.
2.  Assign each example to the nearest centroid.
3.  Move the centroids to the center of the examples that were assigned to it.
4.  Repeat steps 2 and 3 until the cluster assignments do not change or a
    user-defined tolerance or maximum number of iterations is reached.

The next question is, how do we measure similarity between objects? We can define
similarity as the opposite of distance, and a commonly used distance for clustering
examples with continuous features is the squared Euclidean distance between two
points, x and y, in m-dimensional space.
Based on this Euclidean distance metric, we can describe the k-means algorithm as
a simple optimization problem, an iterative approach for minimizing the within-cluster
sum of squared errors (SSE), which is sometimes also called cluster inertia.
"""

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3,
            init='random',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)
"""
Using the preceding code, we set the number of desired clusters to 3; having to
specify the number of clusters a priori is one of the limitations of k-means. We
set n_init=10 to run the k-means clustering algorithms 10 times independently,
with different random centroids to choose the final model as the one with the
lowest SSE. Via the max_iter parameter, we specify the maximum number of iterations
for each single run (here, 300). Note that the k-means implementation in scikit-learn
stops early if it converges before the maximum number of iterations is reached.
However, it is possible that k-means does not reach convergence for a particular
run, which can be problematic (computationally expensive) if we choose relatively
large values for max_iter. One way to deal with convergence problems is to choose
larger values for tol, which is a parameter that controls the tolerance with regard
to the changes in the within-cluster SSE to declare convergence. In the preceding
code, we chose a tolerance of 1e-04 (=0.0001).
A problem with k-means is that one or more clusters can be empty. Note that this
problem does not exist for k-medoids or fuzzy C-means, an algorithm that we will
discuss later in this section. However, this problem is accounted for in the current
k-means implementation in scikit-learn. If a cluster is empty, the algorithm will
search for the example that is farthest away from the centroid of the empty cluster.
Then, it will reassign the centroid to be this farthest point.

When we are applying k-means to real-world data using a Euclidean distance metric,
we want to make sure that the features are measured on the same scale and apply
z-score standardization or min-max scaling if necessary.
"""

plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50, c='lightgreen',
            marker='s', edgecolor='black',
            label='Cluster 1')
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50, c='orange',
            marker='o', edgecolor='black',
            label='Cluster 2')
plt.scatter(X[y_km == 2, 0],
            X[y_km == 2, 1],
            s=50, c='lightblue',
            marker='v', edgecolor='black',
            label='Cluster 3')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()

"""
Although k-means worked well on this toy dataset, we still have the drawback of
having to specify the number of clusters, k, a priori. The number of clusters to
choose may not always be so obvious in real-world applications, especially if we
are working with a higher-dimensional dataset that cannot be visualized. The other
properties of k-means are that clusters do not overlap and are not hierarchical,
and we also assume that there is at least one item in each cluster.
"""

"""K-MEANS++ CLUSTERING SCIKITLEARN"""

"""
So far, we have discussed the classic k-means algorithm, which uses a random seed
to place the initial centroids, which can sometimes result in bad clusterings or
slow convergence if the initial centroids are chosen poorly. One way to address
this issue is to run the k-means algorithm multiple times on a dataset and choose
the best-performing model in terms of the SSE.
Another strategy is to place the initial centroids far away from each other via
the k-means++ algorithm, which leads to better and more consistent results than
the classic k-means.
The initialization in k-means++ can be summarized as follows :
1.  Initialize an empty set, M, to store the k centroids being selected.
2.  Randomly choose the first centroid, from the input examples and assign it to M.
3.  For each example, x(i), that is not in M, find the minimum squared distance,
    d(x(i), M)2, to any of the centroids in M.
4.  To randomly select the next centroid, use a weighted probability distribution.
    For instance, we collect all points in an array and choose a weighted random
    sampling, such that the larger the squared distance, the more likely a point
    gets chosen as the centroid.
5.  Repeat steps 3 and 4 until k centroids are chosen.
6.  Proceed with the classic k-means algorithm.
"""
"""
To use k-means++ with scikit-learn’s KMeans object, we just need to set the init
parameter to 'k-means++'. In fact, 'k-means++' is the default argument to the init
parameter, which is strongly recommended in practice. There is two different
approaches (classic k-means via init='random' versus k-means++ via init='k-means++')
for placing the initial cluster centroids.
"""

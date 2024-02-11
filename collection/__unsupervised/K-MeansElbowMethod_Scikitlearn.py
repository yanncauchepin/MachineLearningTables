
import matplotlib.pyplot as plt
import numpy as np

"""K-MEANS CLUSTERING SCIKITLEARN """

from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3,
            init='random',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)


"""
To quantify the quality of clustering, we need to use intrinsic metrics—such as the within-cluster SSE (distortion)—to compare the performance of different k-means clustering models.
Conveniently, we don’t need to compute the within-cluster SSE explicitly when we are using scikit-learn, as it is already accessible via the inertia_ attribute after fitting a KMeans model :
"""

print(f'Distortion: {km.inertia_:.2f}')

"""ELBOW METHOD"""

"""
Based on the within-cluster SSE, we can use a graphical tool, the so-called elbow
method, to estimate the optimal number of clusters, k, for a given task. We can
say that if k increases, the distortion will decrease. This is because the examples
will be closer to the centroids they are assigned to. The idea behind the elbow
method is to identify the value of k where the distortion begins to increase most
rapidly, which will become clearer if we plot the distortion for different values
of k :
"""

distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()
"""
As you can see, the elbow is located at k = 3, so this is supporting evidence that
k = 3 is indeed a good choice for this dataset.
"""

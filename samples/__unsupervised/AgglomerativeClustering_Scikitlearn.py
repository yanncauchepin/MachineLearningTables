import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""RANDOM DATASET"""

np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample([5, 3])*10
df = pd.DataFrame(X, columns=variables, index=labels)

"""AGGLOMERATIVE CLUSTERING SCIKITLEARN"""

"""
There is also an AgglomerativeClustering implementation in scikit-learn, which
allows us to choose the number of clusters that we want to return. This is useful
if we want to prune the hierarchical cluster tree.
By setting the n_cluster parameter to 3, we will now cluster the input examples
into three groups using the same complete linkage approach based on the Euclidean
distance metric as before :
"""

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3,
                             metric='euclidean',
                             linkage='complete')
labels = ac.fit_predict(X)
print(f'Cluster labels: {labels}')

"""
We can see that the first and the fifth examples (ID_0 and ID_4) were assigned to
one cluster (label 1), and the examples ID_1 and ID_2 were assigned to a second
cluster (label 0). The example ID_3 was put into its own cluster (label 2). Overall,
the results are consistent with the results that we observed in the dendrogram.

Let’s now rerun the AgglomerativeClustering using n_cluster=2 in the following
code snippet :
"""
ac = AgglomerativeClustering(n_clusters=2,
                             metric='euclidean',
                             linkage='complete')
labels = ac.fit_predict(X)
print(f'Cluster labels: {labels}')

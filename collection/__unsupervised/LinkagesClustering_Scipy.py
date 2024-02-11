"""AGGLOMERATIVE HIERARCHICAL CLUSTERING"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""LINKAGES CLUSTERING"""

"""
The two standard algorithms for agglomerative hierarchical clustering are single
linkage and complete linkage. Using single linkage, we compute the distances between
the most similar members for each pair of clusters and merge the two clusters for
which the distance between the most similar members is the smallest. The complete
linkage approach is similar to single linkage but, instead of comparing the most
similar members in each pair of clusters, we compare the most dissimilar members
to perform the merge.
Other commonly used algorithms for agglomerative hierarchical clustering include
average linkage and Ward’s linkage. In average linkage, we merge the cluster pairs
based on the minimum average distances between all group members in the two clusters.
In Ward’s linkage, the two clusters that lead to the minimum increase of the total
within-cluster SSE are merged.
"""

"""COMPLETE LINKAGE CLUSTERING"""

"""
Hierarchical complete linkage clustering is an iterative procedure that can be
summarized by the following steps :
1.  Compute a pair-wise distance matrix of all examples.
2.  Represent each data point as a singleton cluster.
3.  Merge the two closest clusters based on the distance between the most dissimilar
    (distant) members.
4.  Update the cluster linkage matrix.
5.  Repeat steps 2-4 until one single cluster remains.
"""

"""RANDOM DATASET"""
"""
The rows represent different observations (IDs 0-4), and the columns are the
different features (X, Y, Z) of those examples.
"""
np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample([5, 3])*10
df = pd.DataFrame(X, columns=variables, index=labels)
df

"""Random Matrix"""
"""
To calculate the distance matrix as input for the hierarchical clustering algorithm,
we will use the pdist function from SciPy’s spatial.distance submodule.
"""
from scipy.spatial.distance import pdist, squareform
row_dist = pd.DataFrame(squareform(
                        pdist(df, metric='euclidean')),
                        columns=labels, index=labels)
row_dist
"""
We calculated the Euclidean distance between each pair of input examples in our dataset based on the features X, Y, and Z. We provided the condensed distance matrix—returned by pdist—as input to the squareform function to create a symmetrical matrix of the pair-wise distances.
Next, we will apply the complete linkage agglomeration to our clusters using the linkage function from SciPy’s cluster.hierarchy submodule, which returns a so-called linkage matrix.
"""
from scipy.cluster.hierarchy import linkage
help(linkage)

"""
Based on the function description, we understand that we can use a condensed distance
matrix (upper triangular) from the pdist function as an input attribute. Alternatively,
we could also provide the initial data array and use the 'euclidean' metric as a
function argument in linkage. However, we should not use the squareform distance
matrix that we defined earlier, since it would yield different distance values
than expected. To sum it up, the three possible scenarios are listed here :
"""
"""
Incorrect approach: Using the squareform distance matrix as shown in the following
code snippet leads to incorrect results :
"""
row_clusters = linkage(row_dist, method='complete', metric='euclidean')
"""
Correct approach: Using the condensed distance matrix as shown in the following
code example yields the correct linkage matrix :
"""
row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
"""
Correct approach: Using the complete input example matrix (the so-called design
matrix) as shown in the following code snippet also leads to a correct linkage
matrix similar to the preceding approach :
"""
row_clusters = linkage(df.values, method='complete', metric='euclidean')

pd.DataFrame(row_clusters,
             columns=['row label 1',
                      'row label 2',
                      'distance',
                      'no. of items in clust.'],
             index=[f'cluster {(i + 1)}' for i in
                    range(row_clusters.shape[0])])

"""
The linkage matrix consists of several rows where each row represents one merge.
The first and second columns denote the most dissimilar members in each cluster,
and the third column reports the distance between those members. The last column
returns the count of the members in each cluster.
"""

"""Dendrogram"""
from scipy.cluster.hierarchy import dendrogram
# make dendrogram black (part 1/2)
# from scipy.cluster.hierarchy import set_link_color_palette
# set_link_color_palette(['black'])
row_dendr = dendrogram(
    row_clusters,
    labels=labels,
    # make dendrogram black (part 2/2)
    # color_threshold=np.inf
)
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()
"""
Such a dendrogram summarizes the different clusters that were formed during the
agglomerative hierarchical clustering; for example, you can see that the examples
ID_0 and ID_4, followed by ID_1 and ID_2, are the most similar ones based on the
Euclidean distance metric.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""RANDOM DATASET"""

np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample([5, 3])*10
df = pd.DataFrame(X, columns=variables, index=labels)

"""Random Matrix"""
from scipy.spatial.distance import pdist, squareform
row_dist = pd.DataFrame(squareform(
                        pdist(df, metric='euclidean')),
                        columns=labels, index=labels)

from scipy.cluster.hierarchy import linkage

row_clusters = linkage(df.values, method='complete', metric='euclidean')

pd.DataFrame(row_clusters,
             columns=['row label 1',
                      'row label 2',
                      'distance',
                      'no. of items in clust.'],
             index=[f'cluster {(i + 1)}' for i in
                    range(row_clusters.shape[0])])

"""HEAT MAP DENDOGRAM"""

"""
In practical applications, hierarchical clustering dendrograms are often used in
combination with a heat map, which allows us to represent the individual values in
the data array or matrix containing our training examples with a color code. In
this section, we will discuss how to attach a dendrogram to a heat map plot and
order the rows in the heat map correspondingly.
However, attaching a dendrogram to a heat map can be a little bit tricky, so let’s
go through this procedure step by step :
"""
"""
1.  We create a new figure object and define the x axis position, y axis position,
    width, and height of the dendrogram via the add_axes attribute. Furthermore,
    we rotate the dendrogram 90 degrees counterclockwise. The code is as follows :
"""
fig = plt.figure(figsize=(8, 8), facecolor='white')
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters,
                       orientation='left')
# note: for matplotlib < v1.5.1, please use
# orientation='right'
"""
2.  Next, we reorder the data in our initial DataFrame according to the clustering
    labels that can be accessed from the dendrogram object, which is essentially a
    Python dictionary, via the leaves key. The code is as follows :
"""
df_rowclust = df.iloc[row_dendr['leaves'][::-1]]
"""
3.  Now, we construct the heat map from the reordered DataFrame and position it
    next to the dendrogram :
"""
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust,
                  interpolation='nearest',
                  cmap='hot_r')
"""
4.  Finally, we modify the aesthetics of the dendrogram by removing the axis ticks
    and hiding the axis spines. Also, we add a color bar and assign the feature
    and data record names to the x and y axis tick labels, respectively :
"""
axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()

"""
As you can see, the order of rows in the heat map reflects the clustering of the
examples in the dendrogram. In addition to a simple dendrogram, the color-coded
values of each example and feature in the heat map provide us with a nice summary
of the dataset.
"""

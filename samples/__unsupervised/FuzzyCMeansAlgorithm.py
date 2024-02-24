"""FUZZY C MEANS ALGORITHM"""

"""
The FCM procedure is very similar to k-means. However, we replace the hard cluster
assignment with probabilities for each point belonging to each cluster. In k-means,
we could express the cluster membership of an example, x, with a sparse vector of
binary values :
x € teta_1 -> w(i,j) = 0
x € teta_2 -> w(i,j) = 1
x € teta_3 -> w(i,j) = 0

In contrast, a membership vector in FCM could be represented as follows :
x € teta_1 -> w(i,j) = 0,1
x € teta_2 -> w(i,j) = 0,85
x € teta_3 -> w(i,j) = 0,05

Here, each value falls in the range [0, 1] and represents a probability of membership
of the respective cluster centroid. The sum of the memberships for a given example
is equal to 1. As with the k-means algorithm, we can summarize the FCM algorithm
in four key steps :
1.  Specify the number of k centroids and randomly assign the cluster memberships
    for each point.
2.  Compute the cluster centroids.
3.  Update the cluster memberships for each point.
4.  Repeat steps 2 and 3 until the membership coefficients do not change or a
    user-defined tolerance or maximum number of iterations is reached.

The objective function of FCM, we abbreviate it as Jm, looks very similar to the
within-cluster SSE that we minimize in k-means.

However, note that the membership indicator, w(i, j), is not a binary value as in
k-means(w(i,j)€[0,1]), but a real value that denotes the cluster membership
probability (w(i,j)€[0,1]). You also may have noticed that we added an additional
exponent to w(i, j) ; the exponent m, any number greater than or equal to one
(typically m = 2), is the so-called fuzziness coefficient (or simply fuzzifier),
which controls the degree of fuzziness. The larger the value of m, the smaller
the cluster membership, w(i, j), becomes, which leads to fuzzier clusters.

The center, teta_j, of a cluster itself is calculated as the mean of all examples
weighted by the degree to which each example belongs to that cluster (w(i,j)).
"""

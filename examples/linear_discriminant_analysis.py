"""
LDA can be used as a technique for feature extraction to increase
computational efficiency and reduce the degree of overfitting due
to the curse of dimensionality in non-regularized models. The general
concept behind LDA is very similar to PCA, but whereas PCA attempts
to find the orthogonal component axes of maximum variance in a dataset,
the goal in LDA is to find the feature subspace that optimizes class
separability.

A linear discriminant would separate the two normal distributed
classes well. Although the linear discriminant shown on a axis could
captures a lot of the variance in the dataset, it could fail as a good
linear discriminant since this axis does not capture enough of the
class-discriminatory information.

Summarize the main steps that are required to perform LDA :

1.  Standardize the d-dimensional dataset (d is the number of
    features).
2.  For each class, compute the d-dimensional mean vector.
3.  Construct the between-class scatter matrix, SB, and the
    within-class scatter matrix, SW.
4.  Compute the eigenvectors and corresponding eigenvalues of the
    matrix.
5.  Sort the eigenvalues by decreasing order to rank the
    corresponding eigenvectors.
6.  Choose the k eigenvectors that correspond to the k largest
    eigenvalues to construct a d×k-dimensional transformation
    matrix, W ; The eigenvectors are the columns of this matrix.
7.  Project the examples onto the new feature subspace using the
    transformation matrix, W.

As we can see, LDA is quite similar to PCA in the sense that we are
decomposing matrices into eigenvalues and eigenvectors, which will
form the new lower-dimensional feature space. However, as mentioned
before, LDA takes class label information into account, which is
represented in the form of the mean vectors computed in step 2.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""DATASET WINE"""

df_wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/wine/wine.data',
    header=None)
"""Or"""
dataset_path = "/home/yanncauchepin/PrivateProjects/ArtificialIntelligence/Datasets/Table/Table_PopularLearning/Wine/wine.data"
df_wine = pd.read_csv(dataset_path, header=None)

df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']

"""Partition Training Test"""
from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

"""FEATURE SCALING"""

"""Standardization with StandardScaler"""
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

"""FEATURE EXTRACTION SUPERVISED"""

"""LINEAR DISCRIMINANT ANALYSIS"""

np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1,4):
    mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
    print(f'MV {label}: {mean_vecs[label - 1]}\n')

d = 13 # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter
print('Within-class scatter matrix: '
      f'{S_W.shape[0]}x{S_W.shape[1]}')

"""
The assumption that we are making when we are computing the scatter
matrices is that the class labels in the training dataset are
uniformly distributed. However, if we print the number of class
labels, we see that this assumption is violated.
"""
print('Class label distribution:', np.bincount(y_train)[1:])

d = 13 # number of features
S_W = np.zeros((d, d))
for label,mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train==label].T)
    S_W += class_scatter
print('Scaled within-class scatter matrix: '
      f'{S_W.shape[0]}x{S_W.shape[1]}')

mean_overall = np.mean(X_train_std, axis=0)
mean_overall = mean_overall.reshape(d, 1)
d = 13 # number of features
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1) # make column vector
    S_B += n * (mean_vec - mean_overall).dot(
    (mean_vec - mean_overall).T)
print('Between-class scatter matrix: '
      f'{S_B.shape[0]}x{S_B.shape[1]}')


eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i])
               for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs,
              key=lambda k: k[0], reverse=True)
print('Eigenvalues in descending order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])

"""
In LDA, the number of linear discriminants is at most c – 1, where
c is the number of class labels, since the in-between scatter matrix,
SB, is the sum of c matrices with rank one or less. We can indeed
see that we only have two nonzero eigenvalues (the eigenvalues
3-13 are not exactly zero, but this is due to the floating-point
arithmetic in NumPy.).

Note that in the rare case of perfect collinearity (all aligned
example points fall on a straight line), the covariance matrix
would have rank one, which would result in only one eigenvector
with a nonzero eigenvalue.
"""

"""
To measure how much of the class-discriminatory information is
captured by the linear discriminants (eigenvectors), let’s plot
the linear discriminants by decreasing eigenvalues, similar to
the explained variance plot that we created in the PCA section.
For simplicity, we will call the content of class-discriminatory
information discriminability.
"""

tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1, 14), discr, align='center',
        label='Individual discriminability')
plt.step(range(1, 14), cum_discr, where='mid',
         label='Cumulative discriminability')
plt.title('The top two discriminants capture 100 percent of the useful information', fontsize = 12, fontweight = 'bold')
plt.ylabel('"Discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
plt.show()

w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
               eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)

"""
Using the transformation matrix W that we created in the previous
subsection, we can now transform the training dataset by
multiplying the matrices.
"""

X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['o', 's', '^']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train==l, 0],
                X_train_lda[y_train==l, 1] * (-1),
                c=c, label= f'Class {l}', marker=m)
plt.title('Wine classes perfectly separable after projecting the data onto the first two discriminants', fontsize = 12, fontweight = 'bold')
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

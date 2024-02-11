"""
 An alternative approach to feature selection for dimensionality
 reduction is feature extraction : summarize the information content of
 a dataset by transforming it onto a new feature subspace of lower
 dimensionality than the original one. Data compression is an important
 topic in machine learning, and it helps us to store and analyze the
 increasing amounts of data that are produced and collected in the
 modern age of technology.

Similar to feature selection, we can use different feature extraction
techniques to reduce the number of features in a dataset. The difference
between feature selection and feature extraction is that we use feature
extraction to transform or project the data onto a new feature space.

In the context of dimensionality reduction, feature extraction can be
understood as an approach to data compression with the goal of maintaining
most of the relevant information. In practice, feature extraction is not
only used to improve storage space or the computational efficiency of
the learning algorithm but can also improve the predictive performance
by reducing the curse of dimensionality—especially if we are working
with non-regularized models.
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


"""FEATURE EXTRACTION UNSUPERVISED"""

"""PRINCIPAL COMPONENT ANALYSIS"""
"""
Principal component analysis (PCA), an unsupervised linear transformation
technique that is widely used across different fields, is most prominently
for feature extraction and dimensionality reduction. Other popular
applications of PCA include exploratory data analysis and the denoising
of signals in stock market trading, and the analysis of genome data and
gene expression levels in the field of bioinformatics.

PCA helps us to identify patterns in data based on the correlation
between features. In a nutshell, PCA aims to find the directions of
maximum variance in high-dimensional data and projects the data onto a
new subspace with equal or fewer dimensions than the original one. The
orthogonal axes (principal components) of the new subspace can be
interpreted as the directions of maximum variance given the constraint
that the new feature axes are orthogonal to each other.

As a result of transforming the original d-dimensional data onto this
new k-dimensional subspace (typically k << d), the first principal
component will have the largest possible variance. All consequent
principal components will have the largest variance given the constraint
that these components are uncorrelated (orthogonal) to the other
principal components—even if the input features are correlated, the
resulting principal components will be mutually orthogonal (uncorrelated).
Note that the PCA directions are highly sensitive to data scaling, and
we need to standardize the features prior to PCA if the features were
measured on different scales and we want to assign equal importance to
all features.

Before looking at the PCA algorithm for dimensionality reduction in more
detail, let’s summarize the approach in a few simple steps:

1.  Standardize the d-dimensional dataset.
2.  Construct the covariance matrix.
3.  Decompose the covariance matrix into its eigenvectors and eigenvalues.
4.  Sort the eigenvalues by decreasing order to rank the corresponding
    eigenvectors.
5.  Select k eigenvectors, which correspond to the k largest eigenvalues,
    where k is the dimensionality of the new feature subspace ().
6.  Construct a projection matrix, W, from the “top” k eigenvectors.
7.  Transform the d-dimensional input dataset, X, using the projection
    matrix, W, to obtain the new k-dimensional feature subspace.
"""

"""
Eigendecomposition, the factorization of a square matrix into so-called
eigenvalues and eigenvectors, is at the core of the PCA procedure
described in this section. The covariance matrix is a special case of a
square matrix : It’s a symmetric matrix, which means that the matrix is
equal to its transpose. When we decompose such a symmetric matrix, the
eigenvalues are real (rather than complex) numbers, and the eigenvectors
are orthogonal (perpendicular) to each other. Furthermore, eigenvalues
and eigenvectors come in pairs. If we decompose a covariance matrix into
its eigenvectors and eigenvalues, the eigenvectors associated with the
highest eigenvalue corresponds to the direction of maximum variance in
the dataset. Here, this “direction” is a linear transformation of the
dataset’s feature columns.

A positive covariance between two features indicates that the features
increase or decrease together, whereas a negative covariance indicates
that the features vary in opposite directions.
The eigenvectors of the covariance matrix represent the principal
components (the directions of maximum variance), whereas the
corresponding eigenvalues will define their magnitude.

Since the manual computation of eigenvectors and eigenvalues is a
somewhat tedious and elaborate task, we will use the linalg.eig function
from NumPy to obtain the eigenpairs of the Wine covariance matrix.
"""
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n', eigen_vals)


"""
Since we want to reduce the dimensionality of our dataset by compressing
it onto a new feature subspace, we only select the subset of the
eigenvectors (principal components) that contains most of the information
(variance). The eigenvalues define the magnitude of the eigenvectors, so
we have to sort the eigenvalues by decreasing magnitude ; We are
interested in the top k eigenvectors based on the values of their
corresponding eigenvalues.

Using the NumPy cumsum function, we can then calculate the cumulative
sum of explained variances, which we will then plot via Matplotlib’s
step function.
"""
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in
           sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
import matplotlib.pyplot as plt
plt.bar(range(1,14), var_exp, align='center',
        label='Individual explained variance')
plt.step(range(1,14), cum_var_exp, where='mid',
         label='Cumulative explained variance')
plt.title('The proportion of the total variance captured by the principal components', fontsize=12, fontweight='bold')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

"""
We should remind ourselves that PCA is an unsupervised method, which
means that information about the class labels is ignored. Whereas a
random forest uses the class membership information to compute the node
impurities, variance measures the spread of values along a feature axis.
"""

"""
We sort the eigenpairs by descending order of the eigenvalues, construct
a projection matrix from the selected eigenvectors, and use the projection
matrix to transform the data onto the lower-dimensional subspace.
"""

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
                for i in range(len(eigen_vals))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

"""
Next, we collect the two eigenvectors that correspond to the two largest
eigenvalues, to capture about 60 percent of the variance in this dataset.
Note that two eigenvectors have been chosen for the purpose of
illustration, since we are going to plot the data via a two-dimensional
scatterplot later in this subsection. In practice, the number of
principal components has to be determined by a tradeoff between
computational efficiency and the performance of the classifier.
"""
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)

"""
Using the projection matrix, we can now transform an example, x
(represented as a 13-dimensional row vector), onto the PCA subspace
(the principal components one and two) obtaining x′, now a
two-dimensional example vector consisting of two new features.
"""
X_train_std[0].dot(w)

"""
Similarly, we can transform the entire 124×13-dimensional training
dataset onto the two principal components by calculating the matrix
dot product.
"""
X_train_pca = X_train_std.dot(w)

colors = ['r', 'b', 'g']
markers = ['o', 's', '^']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0],
                X_train_pca[y_train==l, 1],
                c=c, label=f'Class {l}', marker=m)
plt.title('Data records from the Wine dataset projected onto a 2D feature space via PCA', fontsize=12, fontweight='bold')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


"""FEATURE ASSESSING IN PRINCIPAL COMPONENT ANALYSIS"""

"""
As we learned, via PCA, we create principal components that
represent linear combinations of the features. Sometimes, we are
interested to know about how much each original feature contributes
to a given principal component. These contributions are often
called loadings.

The factor loadings can be computed by scaling the eigenvectors by
the square root of the eigenvalues. The resulting values can then
be interpreted as the correlation between the original features
and the principal component.
"""
loadings = eigen_vecs * np.sqrt(eigen_vals)

fig, ax = plt.subplots()
ax.bar(range(13), loadings[:, 0], align='center')
ax.set_ylabel('Loadings for PC 1')
ax.set_xticks(range(13))
ax.set_xticklabels(df_wine.columns[1:], rotation=90)
plt.title('Feature correlations with the first principal component', fontsize=12, fontweight='bold')
plt.ylim([-1, 1])
plt.tight_layout()
plt.show()

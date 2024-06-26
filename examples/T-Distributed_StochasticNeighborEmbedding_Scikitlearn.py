"""
One nonlinear dimensionality reduction technique that is particularly
worth highlighting is t-distributed stochastic neighbor embedding
(t-SNE) since it is frequently used in literature to visualize
high-dimensional datasets in two or three dimensions.

If we are dealing with nonlinear problems, which we may encounter
rather frequently in real-world applications, linear transformation
techniques for dimensionality reduction, such as PCA and LDA, may
not be the best choice.

The development and application of nonlinear dimensionality reduction
techniques is also often referred to as manifold learning, where a
manifold refers to a lower dimensional topological space embedded in
a high-dimensional space. Algorithms for manifold learning have to
capture the complicated structure of the data in order to project it
onto a lower-dimensional space where the relationship between data
points is preserved.

While nonlinear dimensionality reduction and manifold learning
algorithms are very powerful, we should note that these techniques
are notoriously hard to use, and with non-ideal hyperparameter choices,
they may cause more harm than good. The reason behind this difficulty
is that we are often working with high-dimensional datasets that we
cannot readily visualize and where the structure is not obvious.
Moreover, unless we project the dataset into two or three dimensions
(which is often not sufficient for capturing more complicated
relationships), it is hard or even impossible to assess the quality
of the results. Hence, many people still rely on simpler techniques
such as PCA and LDA for dimensionality reduction.
"""

"""
In a nutshell, t-SNE is modeling data points based on their
pair-wise distances in the high-dimensional (original) feature space.
Then, it finds a probability distribution of pair-wise distances
in the new, lower-dimensional space that is close to the probability
distribution of pair-wise distances in the original space. Or, in
other words, t-SNE learns to embed data points into a
lower-dimensional space such that the pairwise distances in the
original space are preserved.
However, t-SNE is a technique intended for visualization purposes
as it requires the whole dataset for the projection. Since it projects
the points directly (unlike PCA, it does not involve a projection
matrix), we cannot apply t-SNE to new data points.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""DATASET DIGITS"""
from sklearn.datasets import load_digits
digits = load_digits()
"""
The digits are 8×8 grayscale images. The following code plots the
first four images in the dataset, which consists of 1,797 images
in total.
"""

fig, ax = plt.subplots(1, 4)
for i in range(4):
    ax[i].imshow(digits.images[i], cmap='Greys')
plt.show()

digits.data.shape

X_digits = digits.data
y_digits = digits.target

"""FEATURE EXTRACTION"""

"""T-DISTRIBUTED STOCHASTIC NEIGHBOR EMBEDDING WITH SCIKITLEARN"""

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, init='pca', random_state=123)
X_digits_tsne = tsne.fit_transform(X_digits)

"""
We projected the 64-dimensional dataset onto a 2-dimensional space.
We specified init='pca', which initializes the t-SNE embedding using
PCA as it is recommended in the research.

Note that t-SNE includes additional hyperparameters such as the
perplexity and learning rate (often called epsilon), which we
omitted in the example (we used the scikit-learn default values).
In practice, we recommend to explore these parameters as well.
"""

"""DATA VISUALIZATION"""
import matplotlib.patheffects as PathEffects

def plot_projection(x, colors):
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    for i in range(10):
        plt.scatter(x[colors == i, 0],
                    x[colors == i, 1])
    for i in range(10):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])

plot_projection(X_digits_tsne, y_digits)
plt.title('A visualization of how t-SNE embeds the handwritten digits in a 2D feature space', fontsize=12, fontweight = 'bold')
plt.show()

"""
Like PCA, t-SNE is an unsupervised method, and in the preceding
code, we use the class labels y_digits (0-9) only for visualization
purposes via the functions color argument. Matplotlib’s PathEffects
are used for visual purposes, such that the class label is
displayed in the center (via np.median) of data points belonging
to each respective digit.

As we can see, t-SNE is able to separate the different digits
(classes) nicely, although not perfectly. It might be possible to
achieve better separation by tuning the hyperparameters. However,
a certain degree of class mixing might be unavoidable due to
illegible handwriting. For instance, by inspecting individual
images, we might find that certain instances of the number 3
indeed look like the number 9, and so forth.
"""

"""
Decision tree classifiers are attractive models if we care about
interpretability. As the name “decision tree” suggests, we can
think of this model as breaking down our data by making a
decision based on asking a series of questions.

Based on the features in our training dataset, the decision tree
model learns a series of questions to infer the class labels.
Although the concept of a decision tree based on categorical
variables, the same concept applies if our features are real numbers,
like in the Iris dataset.

Using the decision algorithm, we start at the tree root and split
the data on the feature that results in the largest information gain
(IG). In an iterative process, we can then repeat this splitting
procedure at each child node until the leaves are pure. This means
that the training examples at each node all belong to the same class.
In practice, this can result in a very deep tree with many nodes,
which can easily lead to overfitting. Thus, we typically want to prune
the tree by setting a limit for the maximum depth of the tree.

To split the nodes at the most informative features, we need to
define an objective function to optimize via the tree learning
algorithm. Here, our objective function is to maximize the IG
at each split.

The information gain is simply the difference between the impurity
of the parent node and the sum of the child node impurities—the lower
the impurities of the child nodes, the larger the information gain.
"""

"""
The three impurity measures or splitting criteria that are commonly
used in binary decision trees are Gini impurity (IG), entropy (IH),
and the classification error (IE).

The entropy is therefore 0 if all examples at a node belong to the
same class, and the entropy is maximal, with a value of 1, if
we have a uniform class distribution.

The Gini impurity can be understood as a criterion to minimize the
probability of misclassification. Similar to entropy, the Gini
impurity is maximal if the classes are perfectly mixed.

However, in practice, both the Gini impurity and entropy typically
yield very similar results, and it is often not worth spending much
time on evaluating trees using different impurity criteria rather
than experimenting with different pruning cut-offs. In fact, both
the Gini impurity and entropy have a similar shape.

Another impurity measure is the classification error. This is a
useful criterion for pruning, but not recommended for growing a
decision tree, since it is less sensitive to changes in the class
probabilities of the nodes.
"""

import matplotlib.pyplot as plt
import numpy as np

def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))
x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]
plt.title('Entropy values for different class-membership probabilities', fontsize=12, fontweight='bold')
plt.ylabel('Entropy')
plt.xlabel('Class-membership probability p(i=1)')
plt.plot(x, ent)
plt.show()

def gini(p):
    return p*(1 - p) + (1 - p)*(1 - (1-p))
def entropy(p):
    return - p*np.log2(p) - (1 - p)*np.log2((1 - p))
def error(p):
    return 1 - np.max([p, 1 - p])
x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]
fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
                          ['Entropy', 'Entropy (scaled)',
                           'Gini impurity',
                           'Misclassification error'],
                          ['-', '-', '--', '-.'],
                          ['black', 'lightgray',
                           'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.title('The different impurity indices for different class-membership probabilities between 0 and 1', fontsize=12, fontweight='bold')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('impurity index')
plt.show()

"""
Decision trees can build complex decision boundaries by dividing
the feature space into rectangles. However, we have to be careful
since the deeper the decision tree, the more complex the decision
boundary becomes, which can easily result in overfitting.
"""

"""
EXAMPLE OF DECISION TREE SCIKITLEARN
"""

"""DATASET IRIS"""

"""Loading Iris dataset from scikitlearn"""
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

"""Split the dataset into separate training and test datasets"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

"""No Standardization of X"""

"""BUILD MODEL"""

"""Train a decision tree model with a criterion gini impurity and a deep of 4"""
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree_model.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

"""Plot Mutliclass decision regions"""
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, test_idx=None,
                          resolution=0.02):
    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')
    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='none', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='Test set')

plot_decision_regions(X_combined,
                      y_combined,
                      classifier=tree_model,
                      test_idx=range(105, 150))
plt.title('The decision regions boundaries for this two-dimensional analysis', fontsize=12, fontweight='bold')
plt.xlabel('Petal length [cm]')
plt.ylabel('Petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

"""Overview of features splitting"""
from sklearn import tree
feature_names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
tree.plot_tree(tree_model, feature_names=feature_names, filled=True)
plt.show()

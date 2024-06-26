"""
The goal of ensemble methods is to combine different classifiers into a
meta-classifier that has better generalization performance than each individual
classifier alone. For example, assuming that we collected predictions from
10 experts, ensemble methods would allow us to strategically combine those
predictions by the 10 experts to come up with a prediction that was more
accurate and robust than the predictions by each individual expert.

we will focus on the most popular ensemble methods that use the majority voting
principle. Majority voting simply means that we select the class label that
has been predicted by the majority of classifiers, that is, received more than
50 percent of the votes. Strictly speaking, the term “majority vote” refers to
binary class settings only. However, it is easy to generalize the majority
voting principle to multiclass settings, which is known as plurality voting.
(In the UK, people distinguish between majority and plurality voting via the
terms “absolute” and “relative” majority, respectively.)

Using the training dataset, we start by training m different classifiers.
Depending on the technique, the ensemble can be built from different
classification algorithms, for example, decision trees, support vector machines,
logistic regression classifiers, and so on. Alternatively, we can also use the
same base classification algorithm, fitting different subsets of the training
dataset. One prominent example of this approach is the random forest algorithm
combining different decision tree classifiers.

To predict a class label via simple majority or plurality voting, we can combine
the predicted class labels of each individual classifier and select the class
label that received the most votes.
y^ = mode{ C1(x), C2(x), ..., Cm(x)}
In statistics, the mode is the most frequent event or result in a set.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

"""ENSEMBLE CLASSIFIER ERROR"""

from scipy.special import comb
def ensemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier / 2.))
    probs = [comb(n_classifier, k) *
             error**k *
             (1-error)**(n_classifier - k)
             for k in range(k_start, n_classifier + 1)]
    return sum(probs)
ensemble_error(n_classifier=11, error=0.25)
"""
After we have implemented the ensemble_error function, we can compute the
ensemble error rates for a range of different base errors from 0.0 to 1.0 to
visualize the relationship between ensemble and base errors in a line graph.
"""

error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=11, error=error)
              for error in error_range]
plt.plot(error_range, ens_errors,
         label='Ensemble error',
         linewidth=2)
plt.plot(error_range, error_range,
         linestyle='--', label='Base error',
         linewidth=2)
plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
plt.legend(loc='upper left')
plt.grid(alpha=0.5)
plt.show()
"""
As you can see in the resulting plot, the error probability of an ensemble is
always better than the error of an individual base classifier, as long as the
base classifiers perform better than random guessing (error < 0,5)
"""

"""ENSEMBLE CLASSIFIER WEIGHTING LABEL"""

"""
The algorithm that we are going to implement in this section will allow us to
combine different classification algorithms associated with individual weights
for confidence. Our goal is to build a stronger meta-classifier that balances
out the individual classifiers’ weaknesses on a particular dataset.

To better understand the concept of weighting, we will now take a look at a more
concrete example. Let’s assume that we have an ensemble of three base classifiers,
and we want to predict the class label of a given example, x. Two out of three
base classifiers predict the class label 0, and one, C3, predicts that the
example belongs to class 1. If we weight the predictions of each base classifier
equally, the majority vote predicts that the example belongs to class 0.
Now, let’s assign a weight of 0.6 to C3, and let’s weight C1 and C2 by a
coefficient of 0.2.

To translate the concept of the weighted majority vote into Python code, we can
use NumPy’s convenient argmax and bincount functions, where bincount counts the
number of occurrences of each class label. The argmax function then returns the
index position of the highest count, corresponding to the majority class label
(this assumes that class labels start at 0).
"""
np.argmax(np.bincount([0, 0, 1], weights=[0.2, 0.2, 0.6]))

"""ENSEMBLE CLASSIFIER WEIGHTING PROBABILITIES"""

"""
Certain classifiers in scikit-learn can also return the probability of a predicted
class label via the predict_proba method. Using the predicted class probabilities
instead of the class labels for majority voting can be useful if the classifiers
in our ensemble are well calibrated. The modified version of the majority vote
for predicting class labels from probabilities can be written as follows.

To continue with our previous example, let’s assume that we have a binary
classification problem with class labels and an ensemble of three classifiers.
Let’s assume that the classifiers Cj return the following class membership
probabilities for a particular example, x.
"""
ex = np.array([[0.9, 0.1],
               [0.8, 0.2],
               [0.4, 0.6]])
p = np.average(ex, axis=0, weights=[0.2, 0.2, 0.6])
p
np.argmax(p)

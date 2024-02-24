"""
EXAMPLE OF SCIKITLEARN CLASSIFIER STOCHASTIC GRADIENT DESCENT
"""

"""
Scikit-learn also offers alternative implementations via the SGDClassifier class,
which also supports online learning via the partial_fit method. The concept
behind the SGDClassifier class is similar to the stochastic gradient algorithm.

We could initialize the SGD version of the perceptron (loss='perceptron'),
logistic regression (loss='log'), and an SVM with default parameters
(loss='hinge').
"""
from sklearn.linear_model import SGDClassifier

"""Model of Perceptron"""
ppn = SGDClassifier(loss='perceptron')

"""Model of Logistic regression"""
lr = SGDClassifier(loss='log')

"""Model of Support Vector Machine"""
svm = SGDClassifier(loss='hinge')


"""
The classifiers belong to the so-called estimators in scikit-learn,
with an API that is conceptually very similar to the scikit-learn
transformer API. Estimators have a predict method but can also have
a transform method. As you may recall, we also used the fit method
to learn the parameters of a model when we trained those estimators
for classification. However, in supervised learning tasks, we
additionally provide the class labels for fitting the model,
which can then be used to make predictions about new, unlabeled
data examples via the predict method.
"""

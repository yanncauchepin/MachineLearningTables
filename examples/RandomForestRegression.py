import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""RANDOM FORESTS REGRESSION"""
"""
A random forest, which is an ensemble of multiple decision trees, can be understood
as the sum of piecewise linear functions, in contrast to the global linear and
polynomial regression models. In other words, via the decision tree algorithm, we
subdivide the input space into smaller regions that become more manageable.
"""
"""
An advantage of the decision tree algorithm is that it works with arbitrary features
and does not require any transformation of the features if we are dealing with
nonlinear data because decision trees analyze one feature at a time, rather than
taking weighted combinations into account.
We grow a decision tree by iteratively splitting its nodes until the leaves are
pure or a stopping criterion is satisfied. When we used decision trees for classification,
we defined entropy as a measure of impurity to determine which feature split maximizes
the information gain (IG), which can be defined as follows for a binary split.
To use a decision tree for regression, however, we need an impurity metric that is
suitable for continuous variables, so we define the impurity measure of a node, t,
as the MSE instead.
In the context of decision tree regression, the MSE is often referred to as within-node
variance, which is why the splitting criterion is also better known as variance
reduction.
"""

"""DATASET AMES HOUSING"""

columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area',
           'Central Air', 'Total Bsmt SF', 'SalePrice']
df = pd.read_csv('http://jse.amstat.org/v19n3/decock/AmesHousing.txt',
                 sep='\t',
                 usecols=columns)
df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})
df = df.dropna(axis=0)

"""DECISION TREE REGRESSION SCIKITLEARN"""

"""
To see what the line fit of a decision tree looks like, let’s use the DecisionTreeRegressor
implemented in scikit-learn to model the relationship between the SalePrice and
Gr Living Area variables. Note that SalePrice and Gr Living Area do not necessarily
represent a nonlinear relationship, but this feature combination still demonstrates
the general aspects of a regression tree quite nicely.
"""

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)

"""Decision Tree with a depth of 3"""
from sklearn.tree import DecisionTreeRegressor
X = df[['Gr Liv Area']].values
y = df['SalePrice'].values
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)
sort_idx = X.flatten().argsort()
lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('Living area above ground in square feet')
plt.ylabel('Sale price in U.S. dollars')
plt.show()

"""
As you can see in the resulting plot, the decision tree captures the general trend
in the data. And we can imagine that a regression tree could also capture trends
in nonlinear data relatively well. However, a limitation of this model is that it
does not capture the continuity and differentiability of the desired prediction.
In addition, we need to be careful about choosing an appropriate value for the
depth of the tree so as to not overfit or underfit the data; here, a depth of three
seemed to be a good choice.
"""

"""Decision Tree with a depth of 5"""
from sklearn.tree import DecisionTreeRegressor
X = df[['Gr Liv Area']].values
y = df['SalePrice'].values
tree = DecisionTreeRegressor(max_depth=5)
tree.fit(X, y)
sort_idx = X.flatten().argsort()
lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('Living area above ground in square feet')
plt.ylabel('Sale price in U.S. dollars')
plt.show()

"""Decision Tree with Overall Qal"""
from sklearn.tree import DecisionTreeRegressor
X = df[['Overall Qual']].values
y = df['SalePrice'].values
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)
sort_idx = X.flatten().argsort()
lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('Living area above ground in square feet')
plt.ylabel('Sale price in U.S. dollars')
plt.show()

"""RANDOM FORESTS REGRESSION"""

"""
A random forest usually has a better generalization performance than an individual
decision tree due to randomness, which helps to decrease the model’s variance.
Other advantages of random forests are that they are less sensitive to outliers
in the dataset and don’t require much parameter tuning. The only parameter in random
forests that we typically need to experiment with is the number of trees in the
ensemble. The only difference is that we use the MSE criterion to grow the individual
decision trees, and the predicted target variable is calculated as the average
prediction across all decision trees.
"""
"""
Now, let’s use all the features in the Ames Housing dataset to fit a random forest
regression model on 70 percent of the examples and evaluate its performance on the
remaining 30 percent.
"""
target = 'SalePrice'
features = df.columns[df.columns != target]
X = df[features].values
y = df[target].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(
    n_estimators=1000,
    criterion='squared_error',
    random_state=1,
    n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
from sklearn.metrics import mean_absolute_error
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
print(f'MAE train: {mae_train:.2f}')
print(f'MAE test: {mae_test:.2f}')
from sklearn.metrics import r2_score
r2_train = r2_score(y_train, y_train_pred)
r2_test =r2_score(y_test, y_test_pred)
print(f'R^2 train: {r2_train:.2f}')
print(f'R^2 test: {r2_test:.2f}')
"""
Unfortunately, you can see that the random forest tends to overfit the training
data. However, it’s still able to explain the relationship between the target and
explanatory variables relatively well (R2 = 0.85).
For comparison, the linear model was overfitting less but performed worse on the
test set (R2 = 0.75)
"""

x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
ax1.scatter(y_test_pred, y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
ax2.scatter(y_train_pred, y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
ax1.set_ylabel('Residuals')
for ax in (ax1, ax2):
    ax.set_xlabel('Predicted values')
    ax.legend(loc='upper left')
    ax.hlines(y=0, xmin=x_min-100, xmax=x_max+100,
              color='black', lw=2)
plt.tight_layout()
plt.show()
"""
As it was already summarized by the R2 coefficient, you can see that the model fits
the training data better than the test data, as indicated by the outliers in the
y axis direction. Also, the distribution of the residuals does not seem to be
completely random around the zero center point, indicating that the model is not
able to capture all the exploratory information. However, the residual plot indicates
a large improvement over the residual plot of the linear model that we plotted
earlier in this chapter.
Ideally, our model error should be random or unpredictable. In other words, the
error of the predictions should not be related to any of the information contained
in the explanatory variables; rather, it should reflect the randomness of the
real-world distributions or patterns. If we find patterns in the prediction errors,
for example, by inspecting the residual plot, it means that the residual plots
contain predictive information. A common reason for this could be that explanatory
information is leaking into those residuals.
Unfortunately, there is not a universal approach for dealing with non-randomness
in residual plots, and it requires experimentation. Depending on the data that is
available to us, we may be able to improve the model by transforming variables,
tuning the hyperparameters of the learning algorithm, choosing simpler or more
complex models, removing outliers, or including additional variables.
"""

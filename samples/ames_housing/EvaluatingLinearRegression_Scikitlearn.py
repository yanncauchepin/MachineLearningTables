import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
It is crucial to test the model on data that it hasn’t seen during training to
obtain a more unbiased estimate of its generalization performance.
we want to split our dataset into separate training and test datasets, where we
will use the former to fit the model and the latter to evaluate its performance
on unseen data to estimate the generalization performance. Instead of proceeding
with the simple regression model, we will now use all five features in the dataset
and train a multiple regression model :
"""

"""DATASET AMES HOUSING"""

columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area',
           'Central Air', 'Total Bsmt SF', 'SalePrice']
df = pd.read_csv('http://jse.amstat.org/v19n3/decock/AmesHousing.txt',
                 sep='\t',
                 usecols=columns)
df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})
df = df.dropna(axis=0)

"""MULTIVARIATE REGRESSION"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
target = 'SalePrice'
features = df.columns[df.columns != target]
X = df[features].values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

"""RESIDUAL PLOT"""
"""
Since our model uses multiple explanatory variables, we can’t visualize the linear
regression line (or hyperplane, to be precise) in a two-dimensional plot, but we
can plot the residuals (the differences or vertical distances between the actual
and predicted values) versus the predicted values to diagnose our regression model.
Residual plots are a commonly used graphical tool for diagnosing regression models.
They can help to detect nonlinearity and outliers and check whether the errors are
randomly distributed.
"""
x_max = np.max(
    [np.max(y_train_pred), np.max(y_test_pred)])
x_min = np.min(
    [np.min(y_train_pred), np.min(y_test_pred)])
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(7, 3), sharey=True)
ax1.scatter(
    y_test_pred, y_test_pred - y_test,
    c='limegreen', marker='s',
    edgecolor='white',
    label='Test data')
ax2.scatter(
    y_train_pred, y_train_pred - y_train,
    c='steelblue', marker='o', edgecolor='white',
    label='Training data')
ax1.set_ylabel('Residuals')
for ax in (ax1, ax2):
    ax.set_xlabel('Predicted values')
    ax.legend(loc='upper left')
    ax.hlines(y=0, xmin=x_min-100, xmax=x_max+100,\
        color='black', lw=2)
plt.tight_layout()
plt.show()
"""
In the case of a perfect prediction, the residuals would be exactly zero, which
we will probably never encounter in realistic and practical applications. However,
for a good regression model, we would expect the errors to be randomly distributed
and the residuals to be randomly scattered around the centerline. If we see patterns
in a residual plot, it means that our model is unable to capture some explanatory
information, which has leaked into the residuals, as you can see to a degree in
our previous residual plot. Furthermore, we can also use residual plots to detect
outliers, which are represented by the points with a large deviation from the
centerline.
"""

"""MEAN SQUARED ERROR"""
"""
Another useful quantitative measure of a model’s performance is the mean squared error (MSE) that we discussed earlier as our loss function that we minimized to fit the linear regression model.
Similar to prediction accuracy in classification contexts, we can use the MSE for cross-validation and model selection.
Like classification accuracy, MSE also normalizes according to the sample size, n. This makes it possible to compare across different sample sizes.
"""
from sklearn.metrics import mean_squared_error
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f'MSE train: {mse_train:.2f}')
print(f'MSE test: {mse_test:.2f}')
"""
We can see that the MSE on the training dataset is less than on the test set, which
is an indicator that our model is slightly overfitting the training data in this
case. Note that it can be more intuitive to show the error on the original unit
scale (here, dollar instead of dollar-squared), which is why we may choose to
compute the square root of the MSE, called root mean squared error, or the mean
absolute error (MAE), which emphasizes incorrect prediction slightly less.
"""

"""MEAN ABSOLUTE ERROR"""
from sklearn.metrics import mean_absolute_error
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
print(f'MAE train: {mae_train:.2f}')
print(f'MAE test: {mae_test:.2f}')

"""
Based on the test set MAE, we can say that the model makes an error of approximately
$25,000 on average.
When we use the MAE or MSE for comparing models, we need to be aware that these
are unbounded in contrast to the classification accuracy, for example. In other
words, the interpretations of the MAE and MSE depend on the dataset and feature
scaling. For example, if the sale prices were presented as multiples of 1,000
(with the K suffix), the same model would yield a lower MAE compared to a model
that worked with unscaled features.
"""

"""COEFFICIENT OF DETERMINATION R2"""
"""
It may sometimes be more useful to report the coefficient of determination (R2),
which can be understood as a standardized version of the MSE, for better
interpretability of the model’s performance. Or, in other words, R2 is the fraction
of response variance that is captured by the model. The R2 value is defined as :
R2 = 1 - SSE / SST
SSE is the sum of squared errors, which is similar to the MSE but does not include
the normalization by sample size.
SST is the total sum of squares. In other words, SST is simply the variance of
the response.
R2 is indeed just a rescaled version of the MSE.
R2 = 1 - MSE / Variance(y)
"""
from sklearn.metrics import r2_score
train_r2 = r2_score(y_train, y_train_pred)>>> test_r2 = r2_score(y_test, y_test_pred)
print(f'R^2 train: {train_r2:.3f}, {test_r2:.3f}')
"""
For the training dataset, R2 is bounded between 0 and 1, but it can become negative
for the test dataset. A negative R2 means that the regression model fits the data
worse than a horizontal line representing the sample mean. (In practice, this often
happens in the case of extreme overfitting, or if we forget to scale the test set
in the same manner we scaled the training set.) If R2 = 1, the model fits the data
perfectly with a corresponding MSE = 0.
Evaluated on the training data, the R2 of our model is 0.77, which isn’t great but
also not too bad given that we only work with a small set of features. However,
the R2 on the test dataset is only slightly smaller, at 0.75, which indicates that
the model is only overfitting slightly.
"""

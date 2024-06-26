import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""LINEAR REGRESSION GRADIENT DESCENT"""

class LinearRegressionGD:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.array([0.])
        self.losses_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_
    def predict(self, X):
        return self.net_input(X)

"""DATASET AMES HOUSING"""

columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area',
           'Central Air', 'Total Bsmt SF', 'SalePrice']
df = pd.read_csv('http://jse.amstat.org/v19n3/decock/AmesHousing.txt',
                 sep='\t',
                 usecols=columns)
df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})
df = df.dropna(axis=0)

"""EXAMPLE OF LINEAR REGRESSION GD"""

X = df[['Gr Liv Area']].values
y = df['SalePrice'].values
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD(eta=0.1)
lr.fit(X_std, y_std)

"""
Notice the workaround regarding y_std, using np.newaxis and flatten. Most data
preprocessing classes in scikit-learn expect data to be stored in two-dimensional
arrays. In the previous code example, the use of np.newaxis in y[:, np.newaxis]
added a new dimension to the array. Then, after StandardScaler returned the scaled
variable, we converted it back to the original one-dimensional array representation
using the flatten() method for our convenience.
We discussed in Chapter 2 that it is always a good idea to plot the loss as a
function of the number of epochs (complete iterations) over the training dataset
when we are using optimization algorithms, such as GD, to check that the algorithm
converged to a loss minimum (here, a global loss minimum).
"""
plt.plot(range(1, lr.n_iter+1), lr.losses_)
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.show()
"""
As you can see, the GD algorithm converged approximately after the tenth epoch.
Next, let’s visualize how well the linear regression line fits the training data.
To do so, we will define a simple helper function that will plot a scatterplot of
the training examples and add the regression line.
"""
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
lin_regplot(X_std, y_std, lr)
plt.xlabel(' Living area above ground (standardized)')
plt.ylabel('Sale price (standardized)')
plt.show()
"""
Although this observation makes sense, the data also tells us that the living area
size does not explain house prices very well in many cases. We will discuss how
to quantify the performance of a regression model. We will discuss how we can deal
with outliers later.
"""

"""
In certain applications, it may also be important to report the predicted outcome
variables on their original scale. To scale the predicted price back onto the
original price in U.S. dollars scale, we can simply apply the inverse_transform
method of StandardScaler.
We used the previously trained linear regression model to predict the price of a
house with an aboveground living area of 2,500 square feet. According to our model,
such a house will be worth $292,507.07.
"""
feature_std = sc_x.transform(np.array([[2500]]))
target_std = lr.predict(feature_std)
target_reverted = sc_y.inverse_transform(target_std.reshape(-1, 1))
print(f'Sales price: ${target_reverted.flatten()[0]:.2f}')

"""
As a side note, it is also worth mentioning that we technically don’t have to
update the intercept parameter (for instance, the bias unit, b) if we are working
with standardized variables, since the y axis intercept is always 0 in those cases.
"""
print(f'Slope: {lr.w_[0]:.3f}')
print(f'Intercept: {lr.b_[0]:.3f}')

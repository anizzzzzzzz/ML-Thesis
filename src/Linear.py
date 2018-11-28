import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
desired_width=320

pd.set_option('display.width', desired_width)

np.set_printoptions(linewidth=desired_width)

pd.set_option('display.max_columns',10)

dataset = pd.read_csv('data/insurance.csv')

X = dataset.iloc[:, 0:6].values
y = dataset.iloc[:, 6].values

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# Label Encoder for region
le_1 = LabelEncoder()
X[:,5] = le_1.fit_transform(X[:,5])

# Label Encoder for sex
le_2 = LabelEncoder()
# X[:, 1] = (X[:, 1] == 'male')
X[:, 1] = le_2.fit_transform(X[:, 1])

# Label Encoder for smoker
le_3 = LabelEncoder()
# X[:, 4] = (X[:, 4] == 'yes')
X[:, 4] = le_3.fit_transform(X[:, 4])

# one hot encoder for regions
onehotencode = OneHotEncoder(categorical_features=[5])
X = onehotencode.fit_transform(X).toarray()

# Avoiding dummy variable trap
X = X[:, 1:]

# Splitting into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardizing the data
from sklearn.preprocessing import StandardScaler
se = StandardScaler()
X_train_std = se.fit_transform(X_train)
# standardizing test_data with respect to train_data
X_test_std = se.transform(X_test)

# Applying Linear Regression
from sklearn.linear_model import LinearRegression
le = LinearRegression()
le.fit(X_train_std, y_train)

y_pred = le.predict(X_test_std)

# Calculating coefficients, Mean Squared error, Variance score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# The coefficients
print('Coefficients: \n', le.coef_)
# The mean absolute error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, y_pred))
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# The root mean squared error
print("Root Mean squared error: %.2f"
      % np.sqrt(mean_squared_error(y_test, y_pred)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# Scatter plot
plt.scatter(y_test, y_pred)
plt.show()

# Ditribution plot
sns.distplot((y_test-y_pred))
plt.show()
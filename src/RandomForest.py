import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dataset = pd.read_csv('data/insurance.csv')

# ------------ Trying model with all features -----------------------------
X = dataset.iloc[:, 0:6].values
y = dataset.iloc[:, 6].values

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

# -----------------------------------------------------------------------------

# ----------------  Trying model with only sex, bmi and children features--------------
# dataset = dataset.loc[:, ['sex','bmi', 'children', 'charges']]
#
# X = dataset.iloc[:, 0:3].values
# y = dataset.iloc[:, 3].values
#
# # Label Encoder for sex
# le_1 = LabelEncoder()
# X[:, 0] = le_1.fit_transform(X[:, 0])

# -------------------------------------------------------------------------------------------

# ----------------  Trying model with only bmi, children and smoker features--------------
# dataset = dataset.loc[:, ['bmi', 'children', 'smoker', 'charges']]
#
# X = dataset.iloc[:, 0:3].values
# y = dataset.iloc[:, 3].values
#
# # Label Encoder for smoker
# le_1 = LabelEncoder()
# # X[:, 4] = (X[:, 4] == 'yes')
# X[:, 2] = le_1.fit_transform(X[:, 2])

# -------------------------------------------------------------------------------------------

# ----------------  Trying model with only bmi, childrenm smoker and region features--------------
# dataset = dataset.loc[:, ['bmi', 'children', 'smoker', 'region', 'charges']]
#
# X = dataset.iloc[:, 0:4].values
# y = dataset.iloc[:, 4].values
#
# # Label Encoder for smoker
# le_1 = LabelEncoder()
# # X[:, 4] = (X[:, 4] == 'yes')
# X[:, 2] = le_1.fit_transform(X[:, 2])
#
# # Label Encoder for region
# le_2 = LabelEncoder()
# X[:,3] = le_2.fit_transform(X[:,3])
#
# # one hot encoder for regions
# onehotencode = OneHotEncoder(categorical_features=[3])
# X = onehotencode.fit_transform(X).toarray()
#
# # Avoiding dummy variable trap
# X = X[:, 1:]

# -------------------------------------------------------------------------------------------

# splitting into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Random Forest
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(criterion='mse',n_estimators=500, random_state=1, n_jobs=-1)
forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)

# Calculating  Mean Squared error, Variance score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
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

# plotting in graph
import matplotlib.pyplot as plt
import seaborn as sns
# Scatter plot
plt.scatter(y_test, y_pred)
plt.show()

# Ditribution plot
sns.distplot((y_test-y_pred))
plt.show()
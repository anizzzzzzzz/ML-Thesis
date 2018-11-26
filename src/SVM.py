import pandas as pd

dataset = pd.read_csv('data/insurance.csv')

# print(dataset)
X = dataset.iloc[:, 0:6].values
y = dataset.iloc[:, 6].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Label Encoder for sex
le_1 = LabelEncoder()
X[:, 1] = le_1.fit_transform(X[:, 1])

# Label Encoder for smoker
le_2 = LabelEncoder()
X[:, 4] = le_2.fit_transform(X[:, 4])

# Label Encoder for region
le_3 = LabelEncoder()
X[:, 5] = le_3.fit_transform(X[:, 5])

# One Hot Encoder for region
onehotencoder = OneHotEncoder(categorical_features=[5])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding dummy variable trap
X = X[:, 0:8]

# splitting into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardizing data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# # Normalization
# from sklearn.preprocessing import MinMaxScaler
# sc = MinMaxScaler()
# X_train_std = sc.fit_transform(X_train)
# X_test_std = sc.transform(X_test)

# SVM
from sklearn.svm import SVR
svm = SVR(kernel='rbf')
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)

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

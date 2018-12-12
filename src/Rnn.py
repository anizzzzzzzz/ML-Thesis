import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
desired_width=320

pd.set_option('display.width', desired_width)

np.set_printoptions(linewidth=desired_width)

pd.set_option('display.max_columns',10)

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

# Splitting into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardizing the data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
se = MinMaxScaler()
X_train_std = se.fit_transform(X_train)
# standardizing test_data with respect to train_data
X_test_std = se.transform(X_test)

# Important
# Reshaping X_train and X_test into 3D for input of LSTM
X_train_std = np.reshape(X_train_std, (X_train_std.shape[0], X_train_std.shape[1], 1))
X_test_std = np.reshape(X_test_std, (X_test_std.shape[0], X_test_std.shape[1], 1))

# Building ANN
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# Initializing ANN
regressor = Sequential()

# Adding the input layer and the first hidden layer with Dropout
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(rate = 0.2))

# Adding the second hidden layer with Dropout
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Adding the fourth LSTM layer and some Dropout regularization
regressor.add(LSTM(units=50))
regressor.add(Dropout(rate=0.2))

# Adding the output layer
regressor.add(Dense(units=1))

# compiling the ANN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# fitting the ANN with training dataset
regressor.fit(X_train_std, y_train, batch_size=10, epochs=100)

y_pred = regressor.predict(X_test_std)

# Adding dimension to y_test since regressor returns array in (268,1)
y_test = y_test[:, np.newaxis]

# Calculating coefficients, Mean Squared error, Variance score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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

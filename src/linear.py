# # -*- coding: utf-8 -*-
#
# #Basic libs
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# from sklearn.cross_validation import train_test_split
#
# from sklearn.linear_model import LinearRegression
#
# from sklearn.metrics import mean_squared_error, r2_score
#
# from sklearn.metrics import classification_report ,accuracy_score,confusion_matrix\n"
#
# train_data=pd.read_csv('insurance.csv')
#
#
#
# train_data['bmi']=train_data['bmi'].astype(int)
# train_data['charges']=train_data['charges'].astype(int)
# # Binarization Processing of smoker
# train_data['smoker']=train_data['smoker'].map({'yes':1,'no':0})
# train_data['region']=train_data['region'].map({'southwest':0,'northwest':1,'southeast':2,'northeast':3})
# train_data['sex']=train_data['sex'].map({'female':0,'male':1})
# train_data.head(10)
#
# colormap = plt.cm.RdBu
# plt.figure(figsize=(14,12))
# plt.title('Pearson Correlation of Features', y=1.05, size=15)
# sns.heatmap(train_data.astype(float).corr(),linewidths=0.1,vmax=1.0,
#             square=True, cmap=colormap, linecolor='white', annot=True)
#
# train_data.describe()
#
# #other people's way to see the features relationships
# sns.pairplot(train_data, size=2)
# plt.show()
#
# train_data[['age','charges']].groupby(['age'],as_index=False).mean()
# #The high age,the high charges
#
# train_data[['sex','charges']].groupby(['sex'],as_index=False).mean()
# #men have high charges
#
# train_data[['smoker','charges']].groupby(['smoker'],as_index=False).mean()
# #smoker has high charges
#
# train_data[['region','charges']].groupby(['region'],as_index=False).mean()
# #We can see that the region has little influnce of charges;
#
# train_data[['bmi','charges']].groupby(['bmi'],as_index=False).mean()
#
# train_data[['children','charges']].groupby(['children'],as_index=False).mean()
#
#
# reg=linear_model.LinearRegression()
#
#
# train_data, test_data, train_target,test_target = train_test_split(X.values, y.values, test_size=0.2)
#
# reg.coef_
#
# train_data_pred = regr.predict(train_data).ravel()
# test_target_pred = regr.predict(test_target).ravel()
#
# print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred)))
# print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_test_pred)))
#
# a=reg.predict(x_test)
#
# a[0]
#
# y_test
#
# #mean square error
#
#
#
#
#
#
#
#

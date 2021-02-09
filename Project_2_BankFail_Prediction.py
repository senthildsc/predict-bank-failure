'''
File              : Project_2_BankFail_Prediction.py
Name              : Senthilraj Srirangan
Date              : 01/17/2021
Assignment Number : 7.1 Project 2: Project Draft/Milestone 3
Course            : DSC 680 - Applied Data Science
Exercise Details  :
    Build Bank Fail Prediction Model.
'''

# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif

pd.options.display.max_columns = None
pd.options.display.max_rows = None

df = pd.read_excel('Bank failure data.xlsx', engine='openpyxl')

print(df.head())


df_2009 = df[df['Quarter'] ==  '2009Q4']

#largest bank based on 2009q4
print(df_2009.loc[df_2009['Size'].idxmax()])

df_time = df.copy()
df_time.Quarter = pd.to_datetime(df_time.Quarter)
df_time.set_index('Quarter', inplace=True)
df_plot = df_time[['Net Chargeoffs']]
df_plot.groupby('Quarter').mean().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.show()



#correlation matrix and heatmap

df_corr = df.copy()

df_corr['Failed during 2010Q2'] = df_corr['Failed during 2010Q2'].map({'Yes': 1, 'No': 0})

# calculate the correlation matrix
corr = df_corr.corr()
print('correlation matrix\n',corr)

# plot the heatmap
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns)
corr.style.background_gradient()
corr.style.background_gradient().set_precision(2)

plt.show()

#-----------------------------------------------#

df_model = df.copy()

df_model = df_model.fillna(df_model.mean())
df_model['Failed during 2010Q2'] = df_model['Failed during 2010Q2'].map({'Yes': 1, 'No': 0})

#Separate train and predict data
train = df_model[df_model['Quarter'] < '2010']
predict = df_model[df_model['Quarter'] >= '2010']

#Drop all string data
train_cleaned = train.drop(train.select_dtypes(['object']), axis=1)
predict_cleaned = predict.drop(predict.select_dtypes(['object']), axis=1)

#Get train and predict values
x_train = train_cleaned.drop('Failed during 2010Q2', axis = 1).values
y_train = train_cleaned['Failed during 2010Q2'].values

x_test = predict_cleaned.drop('Failed during 2010Q2', axis = 1).values
y_test = predict_cleaned['Failed during 2010Q2'].values


kbest = SelectKBest(f_classif, k=2)
kbest.fit(x_train,y_train)
mask = kbest.get_support()
new_features = []
feature = list(train_cleaned)


for bool, feature in zip(mask, feature):
    if bool:
        new_features.append(feature)

#Top two predictors
print('New Features\n',new_features)

# Predict Model.

#Running logistic regression to predict
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
prediction = logisticRegr.predict_proba(x_test)
prediction = prediction[:,1].tolist()

print(prediction)


#Creating a new column called Prediction in predict dataframe
predict = predict.assign(Prediction=prediction)
#Sorting values in descending order of probability of bank failing and printing the head
print(predict.sort_values(by=['Prediction'], ascending = False).head())


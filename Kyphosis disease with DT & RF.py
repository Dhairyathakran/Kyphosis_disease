# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 08:33:37 2023

@author: dhair
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#************ IMPORTING DATASET  **************

df_kyphosis = pd.read_csv('/Users/dhair/OneDrive/Desktop/kyphosis.csv')
print('Visualize the Dataset here : \n' , df_kyphosis)

#******** Describe the dataset **********

print(df_kyphosis.describe())

#******* Now checking the dataset the null value is present or not *********

print(df_kyphosis.isnull())
print (df_kyphosis.info())

#*********** Visualizing the dataset ************* 

sns.countplot(df_kyphosis['Kyphosis'] , label = 'Num')
plt.show()

#********* applying the OnehotEncoding Here *********** 

encoding = LabelEncoder()
df_kyphosis['Kyphosis'] = encoding.fit_transform(df_kyphosis['Kyphosis'])
print(df_kyphosis)

# ************ Check the result of kyphosis true or false ***************

kyphosis_false = df_kyphosis[df_kyphosis['Kyphosis']==0]
print(kyphosis_false)

kyphosis_true = df_kyphosis[df_kyphosis['Kyphosis']==1]
print(kyphosis_true)

# ********** Check the percentage **********

print('Disease present after the operation : \n' , (len(kyphosis_true)/len(df_kyphosis)*100))
print('Disease not present after the operation : \n' , (len(kyphosis_false)/len(df_kyphosis)*100))

#********** visualizing the correlation between dataset ************

sns.heatmap(df_kyphosis.corr() , annot = True)
plt.show()

#********* Visualizing the relation between age number start with pairplot *****

sns.pairplot(df_kyphosis , hue = 'Kyphosis' , vars = ['Age' , 'Number' , 'Start'])
plt.show()

#************* Creating the X and Y variable ************** 

X = df_kyphosis.drop(['Kyphosis'] , axis = 1)
print(X)

Y = df_kyphosis['Kyphosis']
print(Y)

#******** Training the data and split ********** 

X_train ,X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.3)

#******** Applying the decision tree here **********

classifier = DecisionTreeClassifier()
classifier.fit(X_train , Y_train)

feature_importances = pd.DataFrame(classifier.feature_importances_ , index = X_train.columns,columns=['importance']).sort_values('importance',ascending=False)
print(feature_importances)

#***************Pedicting the Training Set results *************

y_predict_train = classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(Y_train, y_predict_train)
sns.heatmap(cm, annot=True)                             
plt.show()
print(classification_report(Y_train, y_predict_train))

#**************** Predicting the Testing set results *****************

y_predict_test = classifier.predict(X_test)
cm = confusion_matrix(Y_test, y_predict_test)
sns.heatmap(cm, annot=True)
plt.show()

#***************** Classification Report **************
print(classification_report(Y_test, y_predict_test))

#************ Improve the Model ***************

RandomForest = RandomForestClassifier(n_estimators=150)
RandomForest.fit(X_train, Y_train)

#**************** Predicting the Training set results *****************

y_predict_train = RandomForest.predict(X_train)
y_predict_train
cm = confusion_matrix(Y_train, y_predict_train)
sns.heatmap(cm, annot=True)
plt.show()
print(classification_report(Y_train, y_predict_train))

#**************** Predicting the Testing set results *****************

y_predict_test = RandomForest.predict(X_test)
cm = confusion_matrix(Y_test, y_predict_test)
sns.heatmap(cm, annot=True)
plt.show()

#*************** Classification Report ******************

print(classification_report(Y_test, y_predict_test))











































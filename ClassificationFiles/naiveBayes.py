''' Naive Bayes classifier '''


# importing files
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# importing dataset
X = np.load("X_values.npy")
y = np.load("y_values.npy")


# splitting the data set into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)


# classifier regressor
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)


# predicting results of X_test
y_pred=classifier.predict(X_test)


# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred,)


# calculating accuracy
accuracy = (cm[0,0] + cm[1,1]) * 100 / len(y_pred)
print("Accuracy of Naive Bayes:",accuracy,"%")
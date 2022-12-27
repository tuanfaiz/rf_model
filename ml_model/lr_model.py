import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import glob
import os
import pickle

##=================================================================================================##
##=================================================================================================##

# Read the CSV file :
df = pd.read_csv('C:/Users/tuanf/AndroidStudioProjects/Server/csv_files/combined/FilteredCombined_files.csv')
print(df.info())
df = df.dropna()

#Use LabelEncoder to encode single columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.BSSID = le.fit_transform(df.BSSID)
df.NAME = le.fit_transform(df.NAME)
print(df.head)

#Separating the features (x) and the labels (y)
X = df.drop(["SSID", "Unnamed: 0", "NAME"], axis=1)
y = df["NAME"]

#Train the model
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

from sklearn.metrics import accuracy_score

lr=LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print(accuracy_score(y_test,y_pred))

#Save model using pickle format

pickle.dump(lr, open('lrmodel.pkl', 'wb'))

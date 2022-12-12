import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import glob
import os

#Read the path
file_path = "C:/Users/tuanf/AndroidStudioProjects/Server/csv_files"

#List all the files from the directory
file_list = os.listdir(file_path)

#list all csv files only
csv_files = glob.glob(file_path + "/*.csv")

df_concat = pd.concat([pd.read_csv(f) for f in csv_files ], ignore_index=True)
print(df_concat)

df_concat.to_csv('C:/Users/tuanf/AndroidStudioProjects/Server/csv_files/combined/Combined_files.csv')

#Filtering data
anchor_wifi = pd.DataFrame(df_concat, columns= ['BSSID', 'LEVEL', 'SSID', 'NAME'])
print(anchor_wifi)

save_wname = ["3000", "4000", "UNIFI2.4@unifi", "UNIFI5.0@unifi", "IBRAHIM58@unifi"]

rslt_df = anchor_wifi[anchor_wifi['SSID'].isin(save_wname)] 
print(rslt_df)

rslt_df.to_csv('C:/Users/tuanf/AndroidStudioProjects/Server/csv_files/combined/FilteredCombined_files.csv')

##=================================================================================================##
##=================================================================================================##

# Read the CSV file :
df = pd.read_csv('C:/Users/tuanf/AndroidStudioProjects/Server/csv_files/combined/FilteredCombined_files.csv')
print(df.info())
print(df.sample(5, random_state=44))
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=50, max_features="auto", random_state=44)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(accuracy_score(y_test,y_pred))

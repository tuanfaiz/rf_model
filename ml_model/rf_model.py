import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import sys
import warnings
warnings.filterwarnings("ignore")

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

#Save model using pickle format

pickle.dump(rf, open('rfmodel.pkl', 'wb'))

#Scatter plot

sns.set_style("whitegrid");
sns.pairplot(df, hue="NAME", size=3);
plt.show()
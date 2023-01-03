import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import glob
import os
import seaborn as sns
import sys
import warnings
warnings.filterwarnings("ignore")

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




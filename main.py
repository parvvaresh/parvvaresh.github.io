from Preprocessing import preprocessing
from model import LogisticRegression, extraxt_feature, matrix_features, set_features
from load_file import File
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


#load file
path_sad = "/home/alireza/Desktop/projects/sentiments analysis/data/sad.csv"
paht_joy = "/home/alireza/Desktop/projects/sentiments analysis/data/joy.csv"
file = File(path_sad, paht_joy)
df = file.fit()
print("- data loaded")


#Preprocessing file
preprocess = preprocessing()
df_preprocessing = preprocess.fit(df)
print("-- data preprocess")


#extraxt features
features = extraxt_feature(df_preprocessing["Text"])
X_features = features.fit()
print("--- extraxt features")


#create matrix features
matrix = matrix_features(X_features, df_preprocessing["Text"], df_preprocessing["emotin"])
matrix_feature = matrix.fit() 
print("---- create matrix features")


#set features
model = set_features(matrix_feature, df_preprocessing)
x_features = model.fit()
print("----- set features")

x_train, x_test, y_train, y_test = train_test_split(
     x_features, df["emotin"] , test_size=0.3, random_state=123
)
print("------ train test split")




lg = LogisticRegression()
x_train = np.array(x_train)
y_train = np.array(y_train)
lg.fit(x_train, y_train)
print("------- fit model")
y_pred = lg.predict(np.array(x_test))
print(f"accuracy of model : {lg.accuracy(np.array(y_test), y_pred) * 100}")

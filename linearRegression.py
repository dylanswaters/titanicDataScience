from __future__ import print_function

import os
import subprocess

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

print("Survived Prediction")

# read in certain data for checking the answers

df = pd.read_csv("trainingData2.csv")
testDf = pd.read_csv("testData.csv")

answers = []
fareAnswers = []
datafile = open("A5_test.csv", 'r')
datafile.readline()
for line in datafile:
    data = line.split(",")
    answers.append(int(data[1]))
    fareAnswers.append(float(data[10]))

# generate X and Y train

targets = df["Sex"].unique()

map_to_int = {name: n for n, name in enumerate(targets)}
df["Sex"] = df["Sex"].replace(map_to_int)
testDf["Sex"] = testDf["Sex"].replace(map_to_int)

features = ["Pclass", "Fare", "Sex"]

Y_train = df["Survived"]
X_train = df[features]

# Decision Tree for Survived

dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(X_train, Y_train)

X_test = testDf[features]

predictions = dt.predict(X_test)

correctAnswers = 0
totalAnswers = len(answers)
for i in range(0, len(answers)):
    if(predictions[i] == answers[i]):
        correctAnswers += 1

print("Decison Tree:" + str(correctAnswers) + "/" + str(totalAnswers))

# Neural Net for Survived

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, Y_train.values.ravel())

predictions = mlp.predict(X_test)
correctAnswers = 0
totalAnswers = len(answers)
for i in range(0, len(answers)):
    if(predictions[i] == answers[i]):
        correctAnswers += 1

print("Neural Net:" + str(correctAnswers) + "/" + str(totalAnswers))

# Fare prediction

print("Fare Prediction")

# generate new Y_train and X_train

df = pd.read_csv("A5_train.csv")
targets = df["Sex"].unique()
df["Sex"] = df["Sex"].replace(map_to_int)
features = ["Pclass", "Survived", "Sex"]

Y_train = df["Fare"].astype(int)
X_train = df[features]
X_test = testDf[features].astype(int)

# decision tree for fare

dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(X_train, Y_train)

predictions = dt.predict(X_test)
correctAnswers = 0
totalAnswers = len(fareAnswers)
for i in range(0, len(fareAnswers)):
    if(predictions[i] >= fareAnswers[i]-5 and predictions[i] <= fareAnswers[i]+5):
        correctAnswers += 1

print("Decison Tree:" + str(correctAnswers) + "/" + str(totalAnswers))

# Neural Net for fare

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp2 = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp2.fit(X_train, Y_train)

predictions = mlp2.predict(X_test)
correctAnswers = 0
totalAnswers = len(fareAnswers)
for i in range(0, len(fareAnswers)):
    if(predictions[i] >= fareAnswers[i]-5 and predictions[i] <= fareAnswers[i]+5):
        correctAnswers += 1

print("Neural Net:" + str(correctAnswers) + "/" + str(totalAnswers))

# linear regression for fare

lm = linear_model.LinearRegression()
lm.fit(X_train, Y_train)
predictions = lm.predict(X_test)

correctAnswers = 0
totalAnswers = len(fareAnswers)
for i in range(0, len(fareAnswers)):
    if(predictions[i] >= fareAnswers[i]-5 and predictions[i] <= fareAnswers[i]+5):
        correctAnswers += 1

print("Linear Regression:" + str(correctAnswers) + "/" + str(totalAnswers))

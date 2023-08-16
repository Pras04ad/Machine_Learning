import numpy as np # linear algebra
import pandas as pd 
import os

import math
ftest = pd.read_csv("fruits_test.csv",index_col=False)
ftrain = pd.read_csv("fruits_train.csv",index_col=False)
ftest = ftest.drop(columns=['Id'])
ftrain=ftrain.drop(columns=['Id'])
ftest.head()

def distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))
def neighbor(ftrain,test,k):
    distances = []
    for index, row in ftrain.iterrows():
        dist = distance(row[:-1],test)
        distances.append((index,dist))
    distances.sort(key=lambda x: x[1])
    near = []
    for i in range(k):
        near.append(distances[i][0])
    return near
def predict(ftrain,test,k):
    near = neighbor(ftrain, test, k)
    classes = {}
    for i in near:
        label = ftrain.iloc[i][-1]
        if label in classes:
            classes[label] += 1
        else:
            classes[label] = 1
    sorted_classes = sorted(classes.items(),key=lambda x: x[1],reverse=True)
    return sorted_classes[0][0]
def accuracy(ftrain, ftest, k):
    correct = 0
    for index, row in ftest.iterrows():
        prediction = predict(ftrain,row[:-1],k)
        if prediction == row[-1]:
            correct += 1
            print(correct)
    accuracy =(correct/len(ftest))*100.0
    print(f'{accuracy:.2f}%')
k=5
accuracy(ftrain,ftest,k)
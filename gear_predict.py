import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

train=pd.read_csv("")
test = pd.read_csv("")
train = train.drop_duplicates()
test = train.drop_duplicates()

x = train.drop()
y =train[""]
xtrain,xval,ytrain,yval=train_test_split(x,y,test_size=0.2,random_state=42)

s = StandardScaler()
xtrain=s.fit_transform(xtrain)
xval = s.transform(xval)

model = LogisticRegression()
model.fit(xtrain,ytrain)
pred = model.predict(xval)
accuracy = accuracy_score(pred,yval)
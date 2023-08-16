import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


train=pd.read_csv("")
test = pd.read_csv("")
x = train.drop()
y =train[""]
xtrain,xval,ytrain,yval=train_test_split(x,y,test_size=0.2,random_state=42)

models = [
    ('Logistic Regression',LogisticRegression()),
    ('Random Forest',RandomForestClassifier()),
    ('SVM',SVC()),
    ('Gradient Boosting',GradientBoostingClassifier())
]
acc_mod = {}
for name,model in models:
    model.fit(xtrain,ytrain)
    pred = model.predict(xval)
    acc = accuracy_score(yval,pred)
    acc_mod[name]=acc

for name,accuracy in acc_mod.items():
    print(f"{name:20s}{accuracy:.4f}")

best = max(acc_mod,key = acc_mod.get)

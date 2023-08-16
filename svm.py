from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import confusion_matrix  
from sklearn.model_selection import GridSearchCV

train = pd.read_csv("/kaggle/input/wheet-variety-prediction-svm-b2/wheet_train.csv")
test = pd.read_csv("/kaggle/input/wheet-variety-prediction-svm-b2/wheet_test.csv")

xtrain = train.iloc[:,:-1].values
ytrain = train.iloc[:,-1].values
xval = test.values
grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10, 100]
}
classifier = svm.SVC()
find = GridSearchCV(classifier, grid, cv=5)
find.fit(xtrain, ytrain)
best = svm.SVC(**find.best_params_)
best.fit(xtrain, ytrain)
pred = best.predict(xval)

pred = pd.DataFrame(pred)
pred.columns=['Type']
pred.index.name = 'id'
pred.index += 1
pred
pred.to_csv("Sample_submission.csv")
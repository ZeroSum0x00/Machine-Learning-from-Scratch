from sklearn import datasets
from sklearn.model_selection import train_test_split

from metrics.simple_metrics import accuracy
from supervised_learning.random_forest import Random_Forest

data = datasets.load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

model = Random_Forest(n_trees=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accurancy: ", accuracy(y_test, y_pred))

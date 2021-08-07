from sklearn import datasets
from sklearn.model_selection import train_test_split

from metrics.simple_metrics import accuracy
from supervised_learning.decision_tree import Decision_Tree

data = datasets.load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

model = Decision_Tree(max_depth=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accurancy: ", accuracy(y_test, y_pred))


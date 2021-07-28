from sklearn import datasets
from sklearn.model_selection import train_test_split
from supervised_learning.naive_bayes_example import Naive_Bayes
from metrics.simple_metrics import accuracy

X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

model = Naive_Bayes()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(y_test)
print(predictions)
print('accuracy: ', (accuracy(y_test, predictions) * 100))
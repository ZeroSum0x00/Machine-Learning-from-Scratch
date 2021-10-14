from sklearn import datasets
from sklearn.model_selection import train_test_split
from activations.activations import sigmoid
from optimizers.sgd import SGD
from loss.simple_loss import mse
from metrics.simple_metrics import accuracy
from neural_network.neural_network import Neural_Network

X, y = datasets.make_blobs(n_samples=1000, centers=2, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.T, y_test.T

model = Neural_Network([2, 4, 4, 1], activation=sigmoid)
model.compile(optimizer=SGD, loss=mse, metrics=accuracy)
history = model.fit(x=X_train, y=y_train, batch_size=16, epochs=100, validation_data=(X_test, y_test), learning_rate=0.4)
print(history)
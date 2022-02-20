import numpy as np
from tqdm import tqdm
import sklearn
from matplotlib import pyplot as plt
from activations import sigmoid
from optimizers import get_optimizer_by_name
from losses import get_loss_by_name
from metrics import get_metrics_by_name
from activations import get_activation_by_name

class Neural_Network(object):
    def __init__(self, layers, activation=None, seed=0):
        np.random.seed(seed)

        self.layers = layers
        if activation is not None:
            self.activation = get_activation_by_name(activation)
        else:
            self.activation = sigmoid

        self.weights = [np.random.randn(self.layers[i], self.layers[i - 1]) * np.sqrt(1 / self.layers[i - 1]) for i in
                        range(1, len(self.layers))]
        self.bias = [np.random.rand(n, 1) for n in self.layers[1:]]

    def feed_forward(self, input):
        a = input
        pre_activations = []
        activations = [a]
        for w, b in zip(self.weights, self.bias):
            z = np.dot(w, a) + b
            a = self.activation(z).forward()
            pre_activations.append(z)
            activations.append(a)
        return a, pre_activations, activations

    def compute_deltas(self, pre_activations, y_true, y_pred):
        delta_L = self.loss.derivative(y_true, y_pred) * self.activation(pre_activations[-1]).derivative()
        deltas = [0] * (len(self.layers) - 1)
        deltas[-1] = delta_L
        for l in range(len(deltas) - 2, -1, -1):
            delta = np.dot(self.weights[l + 1].transpose(), deltas[l + 1]) * self.activation(pre_activations[l]).derivative()
            deltas[l] = delta
        return deltas

    def backpropagate(self, deltas, activations):
        dW = []
        db = []
        deltas = [0] + deltas
        for l in range(1, len(self.layers)):
            dW_l = np.dot(deltas[l], activations[l - 1].transpose())
            db_l = deltas[l]
            dW.append(dW_l)
            db.append(np.expand_dims(db_l.mean(axis=1), 1))
        return dW, db

    def fit(self, x=None, y=None, batch_size=None, epochs=1, validation_data=None, shuffle=False,
            initial_epoch=0, steps_per_epoch=None, plot_during_train=None, plot_step=10):

        history_train_losses = []
        history_train_accuracies = []
        history_test_losses = []
        history_test_accuracies = []

        if y is not None:
            x_train, y_train = x, y
        else:
            x_train, y_train = x
        if shuffle:
            x_train, y_train = sklearn.utils.shuffle(x_train, y_train, random_state=0)

        x_test, y_test = validation_data

        y_train, y_test = y_train.reshape(1, -1), y_test.reshape(1, -1)

        if steps_per_epoch is None:
            steps_per_epoch = x_train.shape[1] / batch_size

        for epoch in tqdm(range(initial_epoch, epochs)):
            if x_train.shape[1] % batch_size == 0:
                n_batches = int(x_train.shape[1] / batch_size)
            else:
                n_batches = int(x_train.shape[1] / batch_size) - 1

            batches_x = [x_train[:, batch_size * i:batch_size * (i + 1)] for i in range(0, n_batches)]
            batches_y = [y_train[:, batch_size * i:batch_size * (i + 1)] for i in range(0, n_batches)]

            train_losses = []
            train_accuracies = []

            test_losses = []
            test_accuracies = []

            dw_per_epoch = [np.zeros(w.shape) for w in self.weights]
            db_per_epoch = [np.zeros(b.shape) for b in self.bias]

            for step, (batch_x, batch_y) in enumerate(zip(batches_x, batches_y)):
                if step <= int(steps_per_epoch):
                    batch_y_pred, pre_activations, activations = self.feed_forward(batch_x)
                    deltas = self.compute_deltas(pre_activations, batch_y, batch_y_pred)
                    dW, db = self.backpropagate(deltas, activations)
                    for i, (dw_i, db_i) in enumerate(zip(dW, db)):
                        dw_per_epoch[i] += dw_i / batch_size
                        db_per_epoch[i] += db_i / batch_size

                    batch_y_train_pred = self.predict(batch_x)

                    train_loss = self.loss.forward(batch_y, batch_y_train_pred)
                    train_losses.append(train_loss)
                    train_accuracy = self.metrics(batch_y.T, batch_y_train_pred.T)
                    train_accuracies.append(train_accuracy)

                    batch_y_test_pred = self.predict(x_test)

                    test_loss = self.loss.forward(y_test, batch_y_test_pred)
                    test_losses.append(test_loss)
                    test_accuracy = self.metrics(y_test.T, batch_y_test_pred.T)
                    test_accuracies.append(test_accuracy)

            # weight update
            self.weights, self.bias = self.optimizer.caculator(self.weights, self.bias, dw_per_epoch, db_per_epoch)

            history_train_losses.append(np.mean(train_losses))
            history_train_accuracies.append(np.mean(train_accuracies))

            history_test_losses.append(np.mean(test_losses))
            history_test_accuracies.append(np.mean(test_accuracies))

            if epoch % plot_step == 0:
                print(
                    'Epoch {} / {} | train losses: {} | train accuracy: {} | val losses : {} | val accuracy : {} '.format(
                        epoch, epochs, np.round(np.mean(train_losses), 3), np.round(np.mean(train_accuracies), 3),
                        np.round(np.mean(test_losses), 3), np.round(np.mean(test_accuracies), 3)))

                if plot_during_train:
                    self.plot_decision_regions(x_train, y_train, epoch,
                                               np.round(np.mean(train_losses), 4),
                                               np.round(np.mean(test_losses), 4),
                                               np.round(np.mean(train_accuracies), 4),
                                               np.round(np.mean(test_accuracies), 4),
                                               )
                    plt.show()

            self.plot_decision_regions(x_train, y_train, epoch,
                                       np.round(np.mean(train_losses), 4),
                                       np.round(np.mean(test_losses), 4),
                                       np.round(np.mean(train_accuracies), 4),
                                       np.round(np.mean(test_accuracies), 4),
                                       )

        history = {'epochs': epochs,
                   'train_loss': history_train_losses,
                   'train_acc': history_train_accuracies,
                   'test_loss': history_test_losses,
                   'test_acc': history_test_accuracies
                   }
        return history

    def compile(self, optimizer=None, loss=None, metrics=None):
        self.optimizer = get_optimizer_by_name(optimizer)
        self.loss = get_loss_by_name(loss)
        self.metrics = get_metrics_by_name(metrics)


    def predict(self, a):
        for w, b in zip(self.weights, self.bias):
            z = np.dot(w, a) + b
            a = self.activation(z).forward()
        predictions = (a > 0.5).astype(int)
        return predictions

    def plot_decision_regions(self, X, y, iteration, train_loss, val_loss, train_acc, val_acc, res=0.01):
        X, y = X.T, y.T
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                             np.arange(y_min, y_max, res))

        Z = self.predict(np.c_[xx.ravel(), yy.ravel()].T)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.5)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.scatter(X[:, 0], X[:, 1], c=y.reshape(-1), alpha=0.2)
        plt.title('Training iteration: {}'.format(iteration))
        plt.xlabel('train losses: {} | val losses: {} | train acc: {} | val acc: {}'.format(train_loss,
                                                                                        val_loss,
                                                                                        train_acc,
                                                                                        val_acc))

def history_plot(history):
    n = history['epochs']
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    n = 4000
    plt.plot(range(history['epochs'])[:n], history['train_loss'][:n], label='train_loss')
    plt.plot(range(history['epochs'])[:n], history['test_loss'][:n], label='test_loss')
    plt.title('train & test losses')
    plt.grid(1)
    plt.xlabel('epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(history['epochs'])[:n], history['train_acc'][:n], label='train_acc')
    plt.plot(range(history['epochs'])[:n], history['test_acc'][:n], label='test_acc')
    plt.title('train & test accuracy')
    plt.grid(1)
    plt.xlabel('epochs')
    plt.legend()
    plt.show()

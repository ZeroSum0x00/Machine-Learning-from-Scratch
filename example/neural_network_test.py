from sklearn import datasets
from sklearn.model_selection import train_test_split
from neural_network.neural_network import Neural_Network, history_plot
from utils.preprocess import Preprocessing


if __name__ == '__main__':
    X, y = datasets.make_circles(n_samples=1000, noise=0.025)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.T, y_test.T
    train_ds = (X_train, y_train)
    val_ds = (X_test, y_test)

    epochs = 1200
    plot_step = int(epochs / 10)
    batch_size = 16
    steps_per_epoch = int(X_train.shape[1] / batch_size)

    model = Neural_Network([2, 6, 1], activation='sigmoid', seed=0)
    model.compile(optimizer='SGD', loss='mse', metrics='accuracy')
    history = model.fit(train_ds, batch_size=batch_size, epochs=epochs, validation_data=val_ds,
                        steps_per_epoch=steps_per_epoch, plot_during_train=True, plot_step=plot_step)

    pp = Preprocessing()
    pp.saved_data(history, '../saved_weights/neural_network.plk')

    history_plot(history)
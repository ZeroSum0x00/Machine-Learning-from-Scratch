import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from supervised_learning.naive_bayes import MultinomialNB

if __name__ == '__main__':
    data = pd.read_csv('../datasets/IMDB.csv', encoding='ISO-8859-1')
    train_X, test_X, train_y, test_y = train_test_split(data.loc[:, data.columns != 'sentiment'], data['sentiment'], train_size=0.8)

    model = MultinomialNB()
    model.fit(np.squeeze(train_X.values, 1), train_y.values)
    print(model.predict(np.squeeze(test_X.values, 1)))
    print("MSE on testing data", model.evaluate(np.squeeze(test_X.values, 1), test_y.values, metrics='MSE'))


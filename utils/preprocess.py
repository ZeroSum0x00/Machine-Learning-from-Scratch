import io
import pickle
import numpy as np
import pandas as pd

import sklearn
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

class Preprocessing:

    def __init__(self):
        # self.cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        pass

    def get_data(self, data_storage, dict_X=[], dict_y=[], norm=None, norm_range=[0, 1]):
        if isinstance(data_storage, str):
            data_frame = pd.read_csv(data_storage)
            X = data_frame[dict_X].values
            X = np.array(X, dtype=np.float32)
            y = data_frame[dict_y].values
            y = np.array(y, dtype=np.float32)
        elif isinstance(data_storage, sklearn.utils.Bunch):
            data_frame = pd.DataFrame(data_storage.data, columns=data_storage.feature_names)
            data_frame['MEDV'] = data_storage.target
            X = data_frame[dict_X].values
            X = np.array(X, dtype=np.float32)
            y = data_frame[dict_y].values
            y = np.array(y, dtype=np.float32)
        else:
            X, y = data_storage

        if norm == '0':
            X = MinMaxScaler(feature_range=(norm_range[0], norm_range[1])).fit_transform(X)
            y = MinMaxScaler(feature_range=(norm_range[0], norm_range[1])).fit_transform(y)
        elif norm == '1':
            X = MinMaxScaler(feature_range=(norm_range[0], norm_range[1])).fit_transform(X)
        elif norm == '2':
            X = self._normalizer(X, norm_range)
            y = self._normalizer(y, norm_range)
        elif norm == '3':
            X = self._normalizer(X, norm_range)

        return X, y

    def _normalizer(self, data, norm_range):
        for i in range(data.shape[1]):
            min_value = np.min(data[:, i])
            max_value = np.max(data[:, i])
            data[:, i] = (data[:, i] - min_value) / (max_value - min_value)
            """
                mormalizer value in range(0, 1)
                data = (data - min_value) / (max_value - min_value)
            """
            data[:, i] = data[:, i] * (norm_range[1] - norm_range[0]) + norm_range[0]
            """
                mormalizer value range(0, 1) to (min_range, max_range)
                data = data (range(0, 1)) * (max_range - min_range) + min_range
                https://stackoverflow.com/questions/64535187/scaling-data-to-specific-range-in-python
            """

        return data

    def get_label(self, data_storage):
        data_frame = pd.read_csv(data_storage)
        columns = data_frame.columns
        self.label_mean = data_frame[columns[-1]].astype(str)
        self.label_mean = self.label_mean.values

    def match_label(self, label_pred):
        return self.label_mean[int(label_pred)]

    def saved_data(self, data, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

if __name__ == '__main__':
    data = 'data/value.csv'
    # bostom = datasets.load_boston()

    pp = Preprocessing()
    X, y = pp.get_data(data, dict_X=['value1'], dict_y=['value2'], norm='0', norm_range=[-12, 16])
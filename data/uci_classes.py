'''
Those classes are taken from the maf git repository and preprocess various datasets
git: https://github.com/gpapamak/maf.git

datasets: 
POWER: http://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption
GAS: http://archive.ics.uci.edu/ml/datasets/Gas+sensor+array+under+dynamic+gas+mixtures
MINIBOONE: http://archive.ics.uci.edu/ml/datasets/MiniBooNE+particle+identification
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import data as dt
#import util

data_dir = '/nfs/students/winter-term-2019/project_8/rinder/uci_data'

class POWER:

    class Data:

        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):

        trn, val, tst = self.load_data_normalised()

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]
    """
    def show_histograms(self, split):

        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError('Invalid data split')

        util.plot_hist_marginals(data_split.x)
        plt.show()
    """

    def load_data(self):
        path = os.path.join(os.path.dirname(dt.__file__), data_dir + '/power/data.npy')
        return np.load(path)



    def load_data_split_with_noise(self):

        rng = np.random.RandomState(42)

        data = self.load_data()
        rng.shuffle(data)
        N = data.shape[0]

        data = np.delete(data, 3, axis=1)
        data = np.delete(data, 1, axis=1)
        ############################
        # Add noise
        ############################
        # global_intensity_noise = 0.1*rng.rand(N, 1)
        voltage_noise = 0.01*rng.rand(N, 1)
        # grp_noise = 0.001*rng.rand(N, 1)
        gap_noise = 0.001*rng.rand(N, 1)
        sm_noise = rng.rand(N, 3)
        time_noise = np.zeros((N, 1))
        # noise = np.hstack((gap_noise, grp_noise, voltage_noise, global_intensity_noise, sm_noise, time_noise))
        # noise = np.hstack((gap_noise, grp_noise, voltage_noise, sm_noise, time_noise))
        noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
        data = data + noise

        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]

        return data_train, data_validate, data_test


    def load_data_normalised(self):

        data_train, data_validate, data_test = self.load_data_split_with_noise()
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu)/s
        data_validate = (data_validate - mu)/s
        data_test = (data_test - mu)/s

        return data_train, data_validate, data_test


class GAS:

    class Data:

        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):
        file = os.path.join(os.path.dirname(dt.__file__), data_dir + '/gas/ethylene_CO.pickle')
        trn, val, tst = self.load_data_and_clean_and_split(file)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]
    '''
    def show_histograms(self, split):

        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError('Invalid data split')

        util.plot_hist_marginals(data_split.x)
        plt.show()
    '''

    def load_data(self, file):

        data = pd.read_pickle(file)
        # data = pd.read_pickle(file).sample(frac=0.25)
        # data.to_pickle(file)
        data.drop("Meth", axis=1, inplace=True)
        data.drop("Eth", axis=1, inplace=True)
        data.drop("Time", axis=1, inplace=True)
        return data


    def get_correlation_numbers(self, data):
        C = data.corr()
        A = C > 0.98
        B = A.as_matrix().sum(axis=1)
        return B


    def load_data_and_clean(self, file):

        data = self.load_data(file)
        B = self.get_correlation_numbers(data)

        while np.any(B > 1):
            col_to_remove = np.where(B > 1)[0][0]
            col_name = data.columns[col_to_remove]
            data.drop(col_name, axis=1, inplace=True)
            B = self.get_correlation_numbers(data)
        # print(data.corr())
        data = (data-data.mean())/data.std()

        return data


    def load_data_and_clean_and_split(self, file):

        data = self.load_data_and_clean(file).as_matrix()
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data_train = data[0:-N_test]
        N_validate = int(0.1*data_train.shape[0])
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]

        return data_train, data_validate, data_test


class MINIBOONE:

    class Data:

        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):
        file = os.path.join(os.path.dirname(dt.__file__), data_dir + '/miniboone/data.npy')
        trn, val, tst = self.load_data_normalised(file)

        self.trn = self.Data(trn[:,0:-1])
        self.val = self.Data(val[:,0:-1])
        self.tst = self.Data(tst[:,0:-1])

        self.n_dims = self.trn.x.shape[1]
    '''
    def show_histograms(self, split, vars):

        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError('Invalid data split')

        util.plot_hist_marginals(data_split.x[:, vars])
        plt.show()
    '''

    def load_data(self, root_path):
        # NOTE: To remember how the pre-processing was done.
        # data = pd.read_csv(root_path, names=[str(x) for x in range(50)], delim_whitespace=True)
        # print data.head()
        # data = data.as_matrix()
        # # Remove some random outliers
        # indices = (data[:, 0] < -100)
        # data = data[~indices]
        #
        # i = 0
        # # Remove any features that have too many re-occuring real values.
        # features_to_remove = []
        # for feature in data.T:
        #     c = Counter(feature)
        #     max_count = np.array([v for k, v in sorted(c.iteritems())])[0]
        #     if max_count > 5:
        #         features_to_remove.append(i)
        #     i += 1
        # data = data[:, np.array([i for i in range(data.shape[1]) if i not in features_to_remove])]
        # np.save("~/data/miniboone/data.npy", data)

        data = np.load(root_path)
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]

        return data_train, data_validate, data_test


    def load_data_normalised(self, root_path):

        data_train, data_validate, data_test = self.load_data(root_path)
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu)/s
        data_validate = (data_validate - mu)/s
        data_test = (data_test - mu)/s

        return data_train, data_validate, data_test


class HEPMASS:
    """
    The HEPMASS data set.
    http://archive.ics.uci.edu/ml/datasets/HEPMASS
    """

    class Data:

        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):
        path = os.path.join(os.path.dirname(dt.__file__), data_dir + '/hepmass/')
        trn, val, tst = self.load_data_no_discrete_normalised_as_array(path)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]
    '''
    def show_histograms(self, split, vars):

        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError('Invalid data split')

        util.plot_hist_marginals(data_split.x[:, vars])
        plt.show()
    '''

    def load_data(self, path):

        data_train = pd.read_csv(filepath_or_buffer=os.path.join(path, "1000_train.csv"), index_col=False)
        data_test = pd.read_csv(filepath_or_buffer=os.path.join(path, "1000_test.csv"), index_col=False)

        return data_train, data_test


    def load_data_no_discrete(self, path):
        """
        Loads the positive class examples from the first 10 percent of the dataset.
        """
        data_train, data_test = self.load_data(path)

        # Gets rid of any background noise examples i.e. class label 0.
        data_train = data_train[data_train[data_train.columns[0]] == 1]
        data_train = data_train.drop(data_train.columns[0], axis=1)
        data_test = data_test[data_test[data_test.columns[0]] == 1]
        data_test = data_test.drop(data_test.columns[0], axis=1)
        # Because the data set is messed up!
        data_test = data_test.drop(data_test.columns[-1], axis=1)

        return data_train, data_test


    def load_data_no_discrete_normalised(self, path):

        data_train, data_test = self.load_data_no_discrete(path)
        mu = data_train.mean()
        s = data_train.std()
        data_train = (data_train - mu)/s
        data_test = (data_test - mu)/s

        return data_train, data_test


    def load_data_no_discrete_normalised_as_array(self, path):

        data_train, data_test = self.load_data_no_discrete_normalised(path)
        data_train, data_test = data_train.as_matrix(), data_test.as_matrix()

        i = 0
        # Remove any features that have too many re-occurring real values.
        features_to_remove = []
        for feature in data_train.T:
            c = Counter(feature)
            max_count = np.array([v for k, v in sorted(c.iteritems())])[0]
            if max_count > 5:
                features_to_remove.append(i)
            i += 1
        data_train = data_train[:, np.array([i for i in range(data_train.shape[1]) if i not in features_to_remove])]
        data_test = data_test[:, np.array([i for i in range(data_test.shape[1]) if i not in features_to_remove])]

        N = data_train.shape[0]
        N_validate = int(N*0.1)
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]

        return data_train, data_validate, data_test
import numpy as np
from sklearn.neighbors import NearestNeighbors
import warnings
import math


def dst_euc(X, Y):
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    norm2X = np.linalg.norm(X[:, np.newaxis], axis=2)**2
    norm2Y = np.linalg.norm(Y[:, np.newaxis], axis=2)**2

    root = np.sqrt(np.absolute(norm2X + norm2Y.T - 2*np.dot(X, Y.T)))
    return root


def dst_cos(X, Y):
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    normX = np.sqrt((X**2).sum(axis=1))[:, np.newaxis]
    normY = np.sqrt(((Y.T)**2).sum(axis=0))[:, np.newaxis]

    return np.ones(X.shape[0], Y.shape[0]) - np.dot(X, Y.T)/normX/normY.T


class KNN_classifier:
    'k-NearestNeighbors class'
    k = -1  # amount of neighbors we're taking for analysis
    strategy = 'my_own'  # {'my_own', 'brute', 'kd_tree', 'ball_tree'}
    metric = 'euclidean'  # {'euclidean', 'cosine'}
    weights = True  # trigger for weightened method
    test_block_size = -1  # amount of block size to detect neighbors
    # Here go variables for caching method
    caching = False  # trigger for caching option
    c_inds = np.array([])  # cached indices
    c_wmat = np.array([])  # cached weights

    trainX = np.array([])  # vector of our train objects
    trainY = np.array([])  # vector of answers for train objects
    skNN = NearestNeighbors()

    def __init__(self, k=3, strategy='my_own', metric='euclidean',
                 weights=False, test_block_size=-1, caching=False):
        'Default constructor'
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size
        self.caching = caching

        if strategy in ['brute', 'kd_tree', 'ball_tree']:
            self.skNN = NearestNeighbors(n_neighbors=k, algorithm=strategy,
                                         metric=metric)

    def fit(self, X, y):
        'Fit method for kNN model'
        self.trainY = y.astype(np.int64)  # Writing labels down
        if self.strategy == 'my_own':
            self.trainX = X.astype(np.float64)
        else:
            self.skNN.fit(X.astype(np.float64))

    def find_kneighbors(self, X, return_distance=True):
        'Method to find k nearest objects from our train set'
        if self.strategy == 'my_own':
            dists = np.array([], dtype=np.float64)
            indices = np.array([], dtype=np.int64).reshape((0, self.k))
            nbr_dists = np.array([], dtype=np.float64).reshape((0, self.k))

            if self.metric == 'euclidean':
                dists = dst_euc(X, self.trainX)
            else:
                dists = dst_cos(X, self.trainX)

            for dist_vec in dists:
                indices = np.vstack((indices, np.argsort(dist_vec)[:self.k]))

            if return_distance is True:
                for num, idx_vec in enumerate(indices):
                    nbr_dists = np.vstack((nbr_dists, dists[num][idx_vec]))
                return (indices, nbr_dists)
            else:
                return indices

        else:
            nbr_dists, indices = self.skNN.kneighbors(X)
            if return_distance is True:
                return (indices, nbr_dists)
            else:
                return indices

    def predict(self, X, k=-1):  # if caching is on
        'Predict method'
        if k > 0 and k != self.k:
            if self.caching is False:
                raise ValueError("Using custom k value without caching")
            elif k > self.k:
                raise ValueError("Custom k value is bigger than default k")
            elif len(self.c_inds) < 1:
                raise ValueError("Cache is empty - cannot use it for custom k")

            ans_vec = []
            if self.weights:
                for i in range(len(self.c_inds)):
                    bins = np.bincount(self.trainY[self.c_inds[i][:k]],
                                       weights=self.c_wmat[i][:k])
                    ans_vec.append(np.argmax(bins))
            else:
                for i in range(len(self.c_inds)):
                    bins = np.bincount(self.trainY[self.c_inds[i][:k]])
                    ans_vec.append(np.argmax(bins))
            return ans_vec

        indices = np.array([], dtype=np.int64).reshape((0, self.k))
        nbr_dists = np.array([], dtype=np.float64).reshape((0, self.k))

        if self.weights is True:
            if self.test_block_size > 0:
                block_num = math.ceil(X.shape[0] / float(self.test_block_size))
                block_inds = [min(self.test_block_size*i, X.shape[0])
                              for i in range(block_num+1)]

                for i in range(block_num):
                    curr_block = X[block_inds[i]:block_inds[i+1]]
                    tmp_inds, tmp_nbr = self.find_kneighbors(curr_block, True)
                    indices = np.vstack((indices, tmp_inds))
                    nbr_dists = np.vstack((nbr_dists, tmp_nbr))

            else:
                indices, nbr_dists = self.find_kneighbors(X, True)

            w_matrix = np.array([1.0/(x + 0.00001) for x in nbr_dists])
            ans_vec = []
            for i in range(len(indices)):
                ans_vec.append(np.argmax(np.bincount(self.trainY[indices[i]],
                                                     weights=w_matrix[i])))
            if self.caching is True:
                self.c_inds = indices
                self.c_wmat = w_matrix

            return ans_vec

        else:
            if self.test_block_size > 0:
                block_num = math.ceil(X.shape[0] / float(self.test_block_size))
                block_inds = [min(self.test_block_size*i, X.shape[0])
                              for i in range(block_num+1)]

                for i in range(block_num):
                    curr_block = X[block_inds[i]:block_inds[i+1]]
                    tmp_inds = self.find_kneighbors(curr_block, False)
                    indices = np.vstack((indices, tmp_inds))

            else:
                indices = self.find_kneighbors(X, False)

            ans_vec = []
            for i in range(len(indices)):
                ans_vec.append(np.argmax(np.bincount(self.trainY[indices[i]])))

            if self.caching is True:
                self.c_inds = indices

            return ans_vec

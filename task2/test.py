import sklearn
import numpy as np
from sklearn import datasets
from nearest_neighboors import KNN_classifier
from cross_validation import *
import numpy as np

MNISTdataset = datasets.fetch_mldata('MNIST original')

train_data = MNISTdataset['data'][:60000]
train_labels = MNISTdataset['target'][:60000]
indices = np.arange(60000)
np.random.shuffle(indices)
train_data = train_data[indices]
train_labels = train_labels[indices]


test_data = MNISTdataset['data'][60000:]
test_labels = MNISTdataset['target'][60000:]
indices = np.arange(10000)
np.random.shuffle(indices)
test_data = test_data[indices]
test_labels = test_labels[indices]
del MNISTdataset

cv = kfold(len(train_data[:4000]))
acc_score = knn_cross_val_score(train_data[:4000], train_labels[:4000],
                                range(1, 10), 'accuracy', cv, False,
                                'euclidean', 'ball_tree', 1000)
print(acc_score)

import numpy as np
import scipy as sc
import sklearn.utils as skutils
import svm
import nick_svm
import re


def get_accuracy(pred, real):
    real = list(real)
    pred = list(pred)
    my_sum = 0.0
    for idx in range(len(pred)):
        my_sum += float(pred[idx] == real[idx])
    return my_sum/len(real)

X_fst = np.random.randn(10, 2)
X_sec = 5*np.random.randn(10, 2)+20
X_train = np.vstack((X_fst, X_sec))
y_train = np.hstack((np.ones(10), np.zeros(10)-1))
X_train, y_train = skutils.shuffle(X_train, y_train)

X_fst = np.random.randn(2, 2)
X_sec = 5*np.random.randn(2, 2)+20
X_test = np.vstack((X_fst, X_sec))
y_test = np.hstack((np.ones(2), np.zeros(2)-1))
X_test, y_test = skutils.shuffle(X_test, y_test)

N, D = X_train.shape

nsvm = nick_svm.SVM(C=1.0, method='dual', kernel='linear')
nick_pack = nsvm.fit(X_train, y_train.reshape((N, 1)))
y_pred = nsvm.predict(X_test, return_classes=True)
print(y_pred)
print("Accuracy:", get_accuracy(y_pred, y_test))


my_svm = svm.SVM(C=1.0, method='dual', kernel='linear')
my_pack = my_svm.fit(X_train, y_train)
y_pred = my_svm.predict(X_test, return_classes=True)
print(y_pred)
print("Accuracy:", get_accuracy(y_pred, y_test))

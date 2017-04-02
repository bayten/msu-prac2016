import numpy as np
from nearest_neighboors import KNN_classifier


def kfold(n, n_folds=3):
    idx = np.arange(n)
    fold = np.array([])
    fold_sz = int(n/n_folds)

    test_indices = []
    train_indices = []

    for fold_num in range(n_folds):
        fold = idx[fold_num*fold_sz:(fold_num+1)*fold_sz]
        test_indices.append(fold)
        train_indices.append(idx[~np.in1d(idx, fold)])

    return list(zip(train_indices, test_indices))


def get_accuracy(pred, real):
    real = list(real)
    pred = list(pred)
    my_sum = 0.0
    for idx in range(len(pred)):
        my_sum += float(pred[idx] == real[idx])
    return my_sum/len(real)


def get_fscore(pred, real):
    classes = np.unique(real)
    f_score = 0.0

    for class_tag in classes:
        p_tag = (pred == class_tag)
        r_tag = (real == class_tag)

        TP = np.logical_and(p_tag, r_tag).sum()
        if TP < 0.0001:  # it means that precision and recall equal to 0
            continue

        FP = np.logical_and(p_tag, np.logical_xor(p_tag, r_tag)).sum()
        FN = np.logical_and(r_tag, np.logical_xor(p_tag, r_tag)).sum()
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        f_score += 2*precision*recall/(precision+recall)

    return f_score/len(classes)


def knn_cross_val_score(X, y, k_list, score, cv=None, weights=False,
                        metric="euclidean", strategy="brute",
                        test_block_size=1000):
    if cv is None:
        cv = kfold(len(X))
    k_list = sorted(k_list)
    folds_num = len(cv)

    my_knn = KNN_classifier(k_list[-1], strategy, metric, weights,
                            test_block_size, caching=True)

    k_scores = {}
    for k in k_list:
        k_scores[k] = np.zeros(folds_num)

    if score == "accuracy":
        for fold in range(folds_num):
            my_knn.fit(X[cv[fold][0]], y[cv[fold][0]])
            pred = my_knn.predict(X[cv[fold][1]])
            k_scores[k_list[-1]][fold] = get_accuracy(pred, y[cv[fold][1]])
            for k in k_list[:-1]:
                pred = my_knn.predict(X[cv[fold][1]], k)
                k_scores[k][fold] = get_accuracy(pred, y[cv[fold][1]])
        return k_scores
    elif score == "f_score":
        for fold in range(folds_num):
            my_knn.fit(X[cv[fold][0]], y[cv[fold][0]])
            pred = my_knn.predict(X[cv[fold][1]])
            k_scores[k_list[-1]][fold] = get_fscore(pred, y[cv[fold][1]])

            for k in k_list[:-1]:
                pred = my_knn.predict(X[cv[fold][1]], k)
                k_scores[k][fold] = get_fscore(pred, y[cv[fold][1]])
        return k_scores

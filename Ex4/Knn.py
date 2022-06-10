import numpy as np
from sklearn.metrics import accuracy_score


def Lp_space(x, y, p):
    if p == np.Infinity:
        return np.max(np.abs(x - y))
    else:
        return np.power(sum(np.power(np.abs(x - y), p)), 1 / p)


def prediction(neighbors):
    return [1 if sum(neighbors) > 0 else -1]


def fit(train_X, train_y, test_X, k, p):
    y_pred = []

    for x in test_X:
        dist = []

        for idx, y in enumerate(train_X):
            dist.append(Lp_space(x, y, p))
        np.array(dist)

        # Draw K nearest neighbors
        neighbors = np.argsort(dist)[:k]
        y_pred.append(prediction(train_y[neighbors]))

    return np.array(y_pred)


def wrong_pred(y, pred):
    y = y.reshape(len(y), 1)
    return 1 - accuracy_score(y, pred)

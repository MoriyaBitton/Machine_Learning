import pandas as pd
import matplotlib.pyplot as plt

import random
from math import log
import itertools as it

from sklearn.model_selection import train_test_split

from Perceptron import Perceptron
from Adaboost import *


def readData(file_name):
    data = np.genfromtxt(file_name, dtype=str, encoding=None, skip_footer=0)
    X, y = data[:, :2], data[:, -1]
    return X.astype(float), y.astype(float)


def PerceptronDemo():
    print("\nPerceptron Demo\n")

    X, y = readData('data/two_circle.txt')
    classifier = Perceptron()

    weights = classifier.fit(X, y)

    print("\nWeights Output Vector:", weights,
          "\nScore:", round(classifier.score, 2),
          "\nNumber of Iteration:", classifier.epoch)

    # Plot
    x_ = [min(X[:, 0]), max(X[:, 0])]
    b = -weights[0] / weights[2]
    a = -weights[1] / weights[2]
    y_ = [a * coordinate for coordinate in x_] + b

    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g*")
    plt.plot(X[:, 0][y == -1], X[:, 1][y == -1], "r*")
    plt.title("Perceptron Algorithm")
    plt.plot(x_, y_, 'b-')
    plt.show()

def AdaboostDemo():
    print("\nAdaboost Demo\n")

    x, y = readData('data/four_circle.txt')
    hyp_permutations = pd.Series(list(it.combinations(np.unique(np.arange(S_SIZE)), 2)))
    num_rules = 2 * hyp_permutations.size
    s_x_error, t_x_error = 0.0, 0.0

    # every 2nd rule is negative
    rule_directions = np.ones(num_rules)
    rule_directions[0::2] = -1

    for j in range(NUM_ROUNDS):

        # split the data
        s_x, t_x, s_y, t_y = train_test_split(x, y, test_size=0.5, random_state=random.randint(0, 10000))
        s_y = s_y.reshape(1, S_SIZE)
        s_y_mat = np.vstack([s_y] * num_rules)

        # set all points above line with 1 and points below with -1
        hypothesis_pred = np.zeros((num_rules, S_SIZE), dtype=np.float64)
        for index, loc in enumerate(hyp_permutations):
            p_1, p_2 = s_x[loc[0]], s_x[loc[1]]
            m, n = calc_line_eq_params(p_1, p_2)
            idx = 2 * index
            hypothesis_pred[idx] = get_vector_point_labeling(m, n, s_x, rule_directions[idx])
            hypothesis_pred[idx + 1] = get_vector_point_labeling(m, n, s_x, rule_directions[idx + 1])

        # init points and rule weights
        rules_weight = np.zeros(num_rules, dtype=np.float64)
        points_weight = (1 / S_SIZE) * np.ones((num_rules, S_SIZE), dtype=np.float64)
        best_rules_list = []

        for i in range(NUM_ITER):
            # calc error for all rules
            rules_error = (points_weight * (np.not_equal(hypothesis_pred, s_y_mat).astype(int))).sum(axis=1)

            # get best rule, it's error and weight
            best_rule_idx = np.argmin(rules_error)
            best_rule_error = rules_error[best_rule_idx]
            best_rule_weight = 0.5 * log((1 - best_rule_error) / best_rule_error)
            rules_weight[best_rule_idx] = best_rule_weight

            # update points weights
            best_rule = hypothesis_pred[best_rule_idx]
            points_weight *= np.vstack([np.exp(-1 * best_rule_weight * best_rule * s_y)] * num_rules)
            z_iter = np.sum(points_weight[0, :])
            points_weight = points_weight * (1 / z_iter)

            best_rules_list.append(best_rule_idx)

        # calc model error
        s_x_error += model_eval(best_rules_list, rules_weight, s_x, s_y.T, s_x,
                                hyp_permutations, rule_directions) / NUM_ROUNDS
        t_x_error += model_eval(best_rules_list, rules_weight, t_x, t_y, s_x,
                                hyp_permutations, rule_directions) / NUM_ROUNDS

    print('Train set rules error =', list(np.round(s_x_error, 7)))
    print('Test set rules error =', list(np.round(t_x_error, 7)))


if __name__ == '__main__':
    PerceptronDemo()
    AdaboostDemo()

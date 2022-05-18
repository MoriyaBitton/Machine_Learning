import numpy as np
import matplotlib.pyplot as plt

from Perceptron import Perceptron


def readData(file_name):
    data = np.genfromtxt(file_name, dtype=str, encoding=None, skip_footer=0)
    X, y = data[:, :2], data[:, -1]
    return X.astype(float), y.astype(float)


def PerceptronDemo():
    print("\nPerceptron Demo")

    X, y = readData('two_circle.txt')
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
    print("\nAdaboost Demo")


if __name__ == '__main__':
    PerceptronDemo()
    # AdaboostDemo()

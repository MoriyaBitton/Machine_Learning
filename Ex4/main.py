from Knn import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def readData(file_name):
    data = np.genfromtxt(file_name, dtype=str, encoding=None, skip_footer=0)
    X, y = data[:, :2], data[:, -1]
    return X.astype(float), y.astype(float)


def best_parameters(P, K, table):
    res = []
    for (i, j), val in np.ndenumerate(table):
        if val == table.min():
            res.append([P[i], K[j]])
    [print("P = " + str(val[0]), "K = " + str(val[1])) for val in np.array(res)]


def printing(K, P, error, empirical_error):
    for (i, j), _ in np.ndenumerate(error):
        print("Norm = " + str(P[i]),
              "\tNeighbors = " + str(K[j]),
              "\tTrue Error: {:.2f}%".format(error[i][j]),
              "\tEmpirical Error: {:.2f}%".format(empirical_error[i][j]))

    print("\nBest Parameter base on Testing result:")
    best_parameters(P, K, error)

    print("\nBest Parameter base on Training result:")
    best_parameters(P, K, empirical_error)

    print("\nBest Parameter base on Both:")
    best_parameters(P, K, error + empirical_error)


def plotting(K, P, data, title):
    bar_width = 0.25

    # Set position of bars on X axis
    bar_p1 = np.arange(len(K))
    bar_p2 = [x + bar_width for x in bar_p1]
    bar_pInf = [x + bar_width for x in bar_p2]

    bars = np.array([bar_p1, bar_p2, bar_pInf])

    # Plot
    [plt.bar(bars[i], data[i, :], width=bar_width,
             edgecolor='grey', label='P = ' + str(p))
     for i, p in enumerate(P)]

    plt.xlabel('K value', fontweight='bold', fontsize=15)
    plt.ylabel(title, fontweight='bold', fontsize=15)

    plt.xticks([x + bar_width for x in range(len(K))],
               [str(k) for k in K])

    plt.title(title, fontweight='bold')
    plt.legend()

    plt.savefig(title + '.png', bbox_inches='tight', dpi=150)
    plt.show()


def KnnDemo():
    X, y = readData('two_circle.txt')

    print("Data range:",
          "\nX-axis:", "[" + str(X[0].min()) + " ," + str(X[0].max()) + "]",
          "\nY-axis:", "[" + str(X[1].min()) + " ," + str(X[1].max()) + "]\n")

    epoch = 100
    K = np.array([1, 3, 5, 7, 9])
    P = np.array([1, 2, np.Infinity])

    error = np.zeros(shape=(len(P), len(K)))
    empirical_error = np.zeros(shape=(len(P), len(K)))

    for _ in range(epoch):

        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.5, shuffle=True)

        for idx, p in enumerate(P):
            for k, neighbors in enumerate(K):
                y_pred_test = fit(train_X, train_y, test_X, neighbors, p)
                error[idx][k] += wrong_pred(test_y, y_pred_test)

                y_pred_train = fit(train_X, train_y, train_X, neighbors, p)
                empirical_error[idx][k] += wrong_pred(train_y, y_pred_train)

    # Printing
    printing(K, P, error, empirical_error)

    # Plotting
    plotting(K, P, error, "Error")
    plotting(K, P, empirical_error, "Empirical Error")
    plotting(K, P, error + empirical_error, "Total Error")


if __name__ == '__main__':
    KnnDemo()

import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file

path_train = "data/reg.train.txt"
path_test = "data/reg.test.txt"
np.random.seed(seed=10)


def data_to_numpy():
    X_train, y_train = load_svmlight_file(path_train)
    X_train = X_train.toarray().astype(np.float64)
    X_test, y_test = load_svmlight_file(path_test)
    X_test = X_test.toarray().astype(np.float64)
    return X_train, y_train, X_test, y_test

def data_to_numpy_little():
    data = pd.read_csv("data/machine.data", header=None)
    X = np.array(data[[2, 3, 4, 5, 6, 7]])
    Y = np.array(data[[8]])
    return X, Y

def log_out(i, n_estimators, current_predict, Y, lr):
    sys.stderr.write('\rLearning estimator number: ' + str(i) + "/" + str(n_estimators) \
                     + "; MSE error on train dataset: " + str(MSE(current_predict, Y)) \
                     + "; learning rate: " + str(lr))


def gradient_search(Y, X):
    N = len(Y)
    b = np.zeros(X.shape)
    alpha = 0.04
    for i in range(1000):
        error = X * b - Y
        serorr = np.sum(error**2)/len(X)
        sys.stderr.write('\rError of gradient_search: ' + str(serorr))
        gradient = X * (serorr) / N
        b = b - alpha * gradient
    return b


def plot_graphs(trees_num, losses_my, losses_sklearn):
    fig = plt.figure()
    fig.suptitle('MSE от n_estimators', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel('n_estimators')
    ax.set_ylabel("MSE")
    plt.grid()
    ax.plot(trees_num, losses_my, label="My Gradient Boosting")
    ax.plot(trees_num, losses_sklearn, label="Sklearn")
    plt.legend(loc="best")
    plt.fill_between(trees_num, np.array(losses_sklearn) - np.mean(losses_sklearn) * 0.03,
                     np.array(losses_sklearn) + np.mean(losses_sklearn) * 0.03,
                     alpha=0.1,
                     color="g")
    plt.show()


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


class MySGDClassifier():
    def __init__(self, C=1, alpha=0.1, max_epoch=100):
        """
        C - коэф. регуляризации
        alpha - скорость спуска
        max_epoch - максимальное количество эпох
        """
        self.const = C
        self.alpha = alpha
        self.max_epoch = max_epoch
        self.Beta = None

    def fit(self, X, y=None):
        '''
        Обучение модели
        '''
       # X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # нормализуем данные
        X = np.c_[np.ones_like(X[:, 0]), X]  # искусственно создан дополнительный единичный столбец
        self.Total_loss = []
        Beta = np.random.uniform(size=(X.shape[1],))  # коэффициенты
        self.C = np.full((X.shape[1],), self.const)  # Сделал так, потому что С для Beta[0] равен нулю
        self.C[0] = 0
        N = X.shape[0]
        for epoch in range(self.max_epoch):
            shuffle_indexes = np.random.permutation(np.arange(N))
            X, y = X[shuffle_indexes], y[shuffle_indexes]  # перемешиваем данные
            y_hat = X.dot(Beta)
            error = y_hat - y
            loss = np.sum((error ** 2))  # считаем ошибку
            #print(loss)
            #self.Total_loss.append(loss)  # список ошибок по эпохам
            # обновляем коэффициенты
            gradient = X.T.dot(error)  # считаю градиент на одном объекте
            L1 = self.C * np.sign(Beta)  # L1 регуляризация
            Beta -= (self.alpha * gradient + L1)* (1. / N)
        self.Beta = Beta
        return self


    def predict(self, X):
#        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        X = np.c_[np.ones_like(X[:, 0]), X]  # подготовили данные
        y_pred = X.dot(self.Beta)
        return y_pred





def unit_test():
    linear_regression = MySGDClassifier()
    X, Y = data_to_numpy_little()
    #Y = np.ones((3,1))
    #X = np.random.randn(3, 6)
    print(Y.shape)
    linear_regression.fit(X, Y.ravel())
    y_pred = linear_regression.predict(X)
    print(MSE(y_pred, Y))

if __name__ == "__main__":
    unit_test()

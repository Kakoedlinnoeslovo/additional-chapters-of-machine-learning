from the_latest_realise.LRTree import Tree as Tree
from utils import log_out
from utils import gradient_search
from utils import data_to_numpy
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.metrics import mean_squared_error as MSE
import numpy as np
from utils import plot_graphs
from utils import MSE
from utils import data_to_numpy_little

seed = 10
np.random.seed(seed)

class Gradient_Boosting:
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=10, min_samples_split=1, estimators_list = None, current_predict = None):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.estimators_list = estimators_list
        if estimators_list is None:
            self.estimators_list = list()
        self.current_predict = current_predict


    def fit(self, X, Y):

        if self.current_predict is None:
            first_estimator = np.average(Y)
            self.estimators_list.append(first_estimator)
            self.current_predict = first_estimator


        for i in range(len(self.estimators_list), self.n_estimators):
            antigrad = Y - self.current_predict
            new_estimator = Tree(max_depth = self.max_depth,
                               min_samples_split = self.min_samples_split)
            new_estimator.fit(X, antigrad)
            new_estimator_pred = new_estimator.predict(X)
            learning_rate = self.learning_rate -  (self.learning_rate*(i+1)/self.n_estimators)*0.01
            self.current_predict += learning_rate * new_estimator_pred
            self.estimators_list.append(new_estimator)
            log_out(i, self.n_estimators, self.current_predict, Y, learning_rate)
        return self.estimators_list, self.current_predict

    def predict(self, X):
        #y = self.estimators_list[0].predict(X)
        y = self.estimators_list[0]
        for i, estimator in enumerate(self.estimators_list[1:]):
            learning_rate = self.learning_rate - (self.learning_rate * (i + 1) / self.n_estimators) * 0.01
            y += estimator.predict(X) * learning_rate
        return y



def unit_test():
    X_train, Y_train, X_test, Y_test = data_to_numpy()
    #X_train, Y_train, X_test, Y_test = X[:150, :], Y[:150, :], X[150:, :], Y[150:, :]
    losses_my = list()
    print(X_train.shape, Y_train.shape)
    losses_sklearn = list()
    trees_num = [x*10 for x in range(1,11)]
    #mse on train
    estimators_list = None
    current_predict = None
    for i in trees_num:
        algo1 = Gradient_Boosting(n_estimators=i, min_samples_split=4, max_depth=3, estimators_list =estimators_list , current_predict= current_predict)
        estimators_list, current_predict = algo1.fit(X_train, Y_train.ravel())
        mse = MSE(Y_train, algo1.predict(X_train))
        losses_my.append(mse)
        print("\n My MSE: %.4f" % mse)


        algo = GradientBoostingRegressor(n_estimators=i,
                                         criterion="mse",
                                         max_depth=3,
                                         min_samples_split=4)
        algo.fit(X_train, Y_train.ravel())
        mse = MSE(Y_test, algo.predict(X_test))
        losses_sklearn.append(mse)
        print("sklearn MSE: %.4f" % mse)

    plot_graphs(trees_num, losses_my, losses_sklearn)
    #
    # #mse on test
    # losses_my = list()
    # losses_sklearn = list()
    # for i in trees_num:
    #     algo = Gradient_Boosting(n_estimators=i, min_samples_split=2)
    #     algo.fit(X_train, Y_train)
    #     mse = MSE(Y_test, algo.predict(X_test))
    #     losses_my.append(mse)
    #     print("\n My MSE: %.4f" % mse)
    #     algo = GradientBoostingRegressor(n_estimators=i, criterion="mse", max_depth=3, min_samples_split=2)
    #     algo.fit(X_train, Y_train.ravel())
    #     mse = MSE(Y_test, algo.predict(X_test))
    #     losses_sklearn.append(mse)
    #     print("sklearn MSE: %.4f" % mse)
    #
    # plot_graphs(trees_num, losses_my, losses_sklearn)
if __name__ == "__main__":
    unit_test()

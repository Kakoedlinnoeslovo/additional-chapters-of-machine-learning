from sklearn.linear_model import Lasso, LassoCV, LinearRegression
import numpy as np
from utils import data_to_numpy
from utils import data_to_numpy_little
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import time


class Linear_Regression:
    def __init__(self):
        self.model = Lasso(alpha=0.005, max_iter=10000)
    def fit(self, X, Y):
        self.model.fit(X,Y)
    def predict_sample(self, X):
        X = X.reshape(1,-1)
        predict = self.model.predict(X=X)
        return predict
    def predict(self, X):
        #X = X.reshape(1,-1)
        predict = self.model.predict(X=X)
        return predict

class Tree:
    def __init__(self, max_depth = 10, min_samples_split = 4, SD = None, root = True):
        self.model = Linear_Regression()
        self.leaf = False
        self.split_value = None
        self.split_feature = None
        self.min_samples_split = min_samples_split
        self.depth = max_depth
        if root:
            self.max_depth = max_depth
        else:
            self.max_depth = -1#children
        self.SD = SD



    def fit(self, X, y):

        sum_squared = np.sum(y ** 2)
        sum = np.sum(y)
        n_all = len(y)

        node_mse = global_mse = self._mse_metric(sum_squared=sum_squared, sum=sum, n_all=n_all)


        if self.depth == self.max_depth:
            self.SD = global_mse


        if len(y) < self.min_samples_split or self.depth == 0 or global_mse< 0.05 * self.SD:
            self.split_value = None
            self.split_feature = None
            self.leaf = True
            self.model.fit(X,y)
            self.data = np.sum(y) / len(y)
            return self

        best_mse = None
        best_split_value = None
        best_split_feature = None
        best_split_index = None
        best_sorted_indexes = None

        #print(y.shape)

        for current_split_feature in range(0, X.shape[1], 1):
            coloumn = X[:,current_split_feature]
            sorted_indexes = np.argsort(coloumn)
            current_mse, current_split_value, current_split_index = self._find_best_split(X,
                                                                     y,
                                                                     sorted_indexes,
                                                                     current_split_feature,
                                                                     global_mse)
            if current_split_value is None:
                continue
            if current_mse < global_mse:
                global_mse = current_mse
                best_mse = global_mse
                best_split_value = current_split_value
                best_split_feature = current_split_feature
                best_sorted_indexes = sorted_indexes
                best_split_index = current_split_index


        #if global_mse < 0.05 * self.SD:#todo don't forget to send this info to the last node
        if best_mse is not None:

            self.split_value = best_split_value
            self.split_feature  = best_split_feature
            X_l, X_r, y_l, y_r = X[best_sorted_indexes][:best_split_index], \
                                 X[best_sorted_indexes][best_split_index:], \
                                 y[best_sorted_indexes][:best_split_index], \
                                 y[best_sorted_indexes][best_split_index:]

            if len(X_l) == 0 or len(X_r) ==0:
                self.split_value = None
                self.split_feature = None
                self.leaf = True
                self.model.fit(X, y)
                self.data = np.sum(y) / len(y)
                return self

            if X_l.shape[0] < self.min_samples_split or best_mse < 0.05 * self.SD:
                self.left_tree = Tree(max_depth=0,
                                      min_samples_split=self.min_samples_split,
                                      SD = self.SD, root = False).fit(X_l, y_l)

            else:
                self.left_tree = Tree(max_depth=self.depth - 1,
                                     min_samples_split=self.min_samples_split,
                                     SD = self.SD, root=False).fit(X_l, y_l)

            if X_r.shape[0] < self.min_samples_split or best_mse < 0.05 * self.SD:
                self.right_tree = Tree(max_depth=0,
                                       min_samples_split=self.min_samples_split,
                                       SD = self.SD, root=False).fit(X_r, y_r)
            else:
                self.right_tree = Tree(max_depth=self.depth - 1,
                                       min_samples_split=self.min_samples_split,
                                       SD=self.SD, root=False).fit(X_r, y_r)
        else:
            self.split_value = None
            self.split_feature = None
            self.model.fit(X, y)
            self.data = np.sum(y) / len(y)
            self.leaf = True
            return self
        return self


    def _find_best_split(self, X, y, sorted_indexes,  split_feature, global_mse):

        best_mse = global_mse
        best_split_value  = None
        best_split_index = None

        sorted_y = y[sorted_indexes]
        sorted_X = X[sorted_indexes]

        s_l = 0
        n_l = 0
        ss_l = 0


        s_r = np.sum(sorted_y)
        n_r = len(sorted_y)
        ss_r = np.sum(sorted_y**2)



        for split_index in range(0, sorted_X.shape[0]-1):
            temp_X = sorted_X[split_index, split_feature]
            temp_y = sorted_y[split_index]
            temp_X_next = sorted_X[split_index+1, split_feature]

            n_l += 1
            s_l += temp_y
            ss_l+=temp_y**2

            n_r -=1
            s_r -= temp_y
            ss_r-=temp_y**2

            if temp_X_next > temp_X:
                mse_l = self._mse_metric(ss_l, s_l, n_l)
                mse_r = self._mse_metric(ss_r, s_r, n_r)
                new_mse = mse_l +mse_r
                if new_mse < best_mse:
                    best_mse = new_mse
                    best_split_value = (temp_X_next + temp_X)/2
                    best_split_index = split_index
        return best_mse, best_split_value, best_split_index


    def _mse_metric(self, sum_squared, sum, n_all):
        mse_metric = (sum_squared) - (sum ** 2) / n_all
        return mse_metric


    def _predict_one(self, sample):
        if (self.leaf):
            value = np.float(self.model.predict_sample(sample))
            value_sr = self.data
            return value
        if sample[self.split_feature]>= self.split_value:
            return self.right_tree._predict_one(sample)
        else:
            return self.left_tree._predict_one(sample)

    def predict(self, X):
        predicted = np.zeros(X.shape[0],)
        for i,sample in enumerate(X):
            predicted[i] = self._predict_one(sample)
        return predicted




def unit_test():
    X, Y, X_t, Y_t = data_to_numpy()
    #X = np.array([1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]).reshape(-1, 1)
    #Y = np.array([0.1, 0.2, 0.5, 0.6, 0.7, 0.6, 0.5, 0.2, 0.1]).ravel()
    # X_train, Y_train, X_test, Y_test = X[:150,:], Y[:150,:], X[150:180,:], Y[150:180,:]
    tree = Tree(max_depth=3, min_samples_split=4)
    tree.fit(X, Y.ravel())
    #tree.print_tree()

    Y_pred = tree.predict(X)
    print(mean_squared_error(Y, Y_pred))
    tree = DecisionTreeRegressor(max_depth=3, min_samples_split=4)
    tree.fit(X, Y.ravel())
    Y_pred = tree.predict(X)
    print(mean_squared_error(Y, Y_pred))

if __name__ == "__main__":
    unit_test()
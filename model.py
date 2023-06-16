import pandas as pd
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def accuracy(self, y, y_pred):
    	return np.sum(y == y_pred) / y.shape[0]

class extraxt_feature:
	def __init__(self , x):
		self.x = x

	def fit(self):
		result = []
		for words in self.x:
			result.extend(words)
		return list(set(result))


class matrix_features:
	def __init__(self, words, X, target):
		self.target = target
		self.X = X
		self.df = pd.DataFrame()
		self.df["words"] = words
		self.df.set_index("words", inplace = True)
		self.df["freq postive"] = [0 for _ in range(len(words))]
		self.df["freq negetive"] = [0 for _ in range(len(words))]

	def fit(self):
		for index in range(self.X.shape[0]):
			words = self.X.iloc[index]
			for word in words:
				if self.target.iloc[index] == 1:
					self.df.loc.__setitem__(([word], "freq postive"), self.df.loc[word]["freq postive"] + 1)

				if self.target.iloc[index] == 0:
					self.df.loc.__setitem__(([word], "freq negetive"), self.df.loc[word]["freq negetive"] + 1)
		return self.df


class set_features:
	def __init__(self, df, x):
		self.df = df
		self.x = x
		self.X_test = pd.DataFrame()

	def fit(self):
		result_sum_postive = []
		result_sum_negetive = []
		for index in range(self.x.shape[0]):
			words = self.x.iloc[index]["Text"]
			sum_postive = 0
			sum_negetive = 0
			for word in words:
				sum_postive += self.df.loc[word]["freq postive"]
				sum_negetive += self.df.loc[word]["freq negetive"]
			result_sum_negetive.append(sum_negetive)
			result_sum_postive.append(sum_postive)
		self.X_test["sum postive"] = result_sum_postive
		self.X_test["sum negetive"] = result_sum_negetive
		return self.X_test
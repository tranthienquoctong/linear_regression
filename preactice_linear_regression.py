'''CopyRight by VanLoc!!!'''
import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

def read_data(path, filename):
	path_file = os.path.join(path, filename)
	df = pd.read_csv(path_file)
	return df

def extract_feature_label(df):
	X = df.iloc[:, :-1]
	y = df.iloc[:, -1:]
	return X.values, y.values

def cost_caculate(y, y_predict):
	r, c = np.shape(y)
	return np.sum(np.power(y_predict - y, 2)) / r

def fit(X, y, learning_rate = 0.0001, iteractions = 50):
	r, c = np.shape(X)
	one = np.ones((r, 1))
	X = np.concatenate((one, X), axis = 1)
	W_init = np.random.rand(c + 1 , 1)
	w = [W_init]
	cost_histories = []
	for iteraction in range(iteractions):
		y_predict = np.dot(X, w[-1])
		loss = y_predict - y
		gradient = learning_rate * (np.dot(X.T, loss) / r)
		w_new = w[-1] - gradient
		w.append(w_new)
		cost = cost_caculate(y, y_predict)
		cost_histories.append(cost)
	return w[-1], cost_histories

def draw_cost_histories(cost_histories):
	plt.xlabel('X')
	plt.ylabel('y')
	histories = len(cost_histories)
	plt.plot(range(histories), cost_histories)
	plt.show()


if __name__ == '__main__':
	path = 'data'
	filename = 'data.csv'
	df = read_data(path, filename)
	print(df)
	X, y = extract_feature_label(df)
	w, cost_histories = fit(X, y)
	draw_cost_histories(cost_histories)

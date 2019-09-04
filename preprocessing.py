import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def read_data(root, filename):
	path_file = os.path.join(root, filename)
	df = pd.read_csv(path_file)
	return df 

def extract_label_and_feauture(df):
	y = df.iloc[:,-1:]
	X = df.iloc[:, :1]
	return  X.values, y.values

def hypothesis(X, w):
	return np.dot(X, w)

def cost_caculate(y, y_predict):
	r = np.shape(y)[0]
	return np.sum(np.power((y - y_predict), 2)) / r

def fit(X, y, learning_rate = 0.0001, iterations = 100):
	r, c = np.shape(X)
	one = np.ones((r, 1))
	X = np.concatenate((X, one), axis= 1)
	W_init = np.random.rand(1, c + 1).T
	w = [W_init]
	cost_history = []
	for iteration in range(iterations):
		y_predict = hypothesis(X, w[-1])
		loss = y_predict - y 
		gradient = learning_rate * np.dot(X.T, loss) / r
		w_new = w[-1] - gradient
		cost = cost_caculate(y, y_predict)
		cost_history.append(cost)
		w.append(w_new)
	return w, cost_history

def predict(X_test, w):
	return (w[0][0] + x*w[0][1] for x in X_test)

def draw_cost(cost_histories):
	plt.xlabel('x')
	plt.ylabel('y')
	iterations = len(cost_histories)
	plt.plot(range(iterations), cost_histories, 'b.')
	plt.show()

if __name__ == '__main__':
	root = 'data'
	df = read_data(root, 'data.csv')
	X, y = extract_label_and_feauture(df)
	w, cost_history = fit(X, y)
	draw_cost(cost_history)

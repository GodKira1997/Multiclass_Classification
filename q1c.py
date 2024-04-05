"""
file: q1c.py
description: Problem 1c
language: python3
author: Anurag Kallurwar, ak6491@rit.edu
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Meta-Parameters
beta = 0.001 # regularization coefficient
alpha = 0.01 # step size coefficient
n_epoch = 10000 # number of epochs (full passes through the dataset)
eps = 0.00001 # controls convergence criterion
m = 5 # Size of mini-batch


def create_mini_batch(X, y):
	"""
	Create mini-batches of training data by rnadomly drawing a vector from
	the Dataset with no replacement
	:param X: Input matrix NxD
	:param y: output y NxK
	:return: Dataset or batch of mxK
	"""
	vector_indices = np.arange(len(X))
	mini_batches = []
	used_vectors = []
	while len(used_vectors) < len(X):
		mask = np.isin(vector_indices, used_vectors)
		unused_vectors = vector_indices[~mask]
		indices = np.random.choice(unused_vectors, m, replace=False)
		X_batch = X[indices]
		y_batch = y[indices]
		used_vectors += list(indices)
		mini_batches.append((X_batch, y_batch))
	return mini_batches


def encode_one_hot(y, K):
	"""
	Encoding labels to one hot coding => Nx1 to NxK
	:param y: Original output y Nx1
	:param K: Number of classes
	:return: encoded labels NxK
	"""
	encoded_y = []
	for value in y:
		one_hot_encode = [0] * K
		one_hot_encode[value[0]] = 1
		encoded_y.append(one_hot_encode)
	return np.array(encoded_y)


def decode_one_hot(y):
	""""
	Decoding one hot coding to labels => NxK to Nx1
	:param y: Encoded output y NxK
	:return: decoded labels y Nx1
	"""
	y_pred_decoded = np.zeros((y.shape[0], 1), dtype=int)
	for i in range(y.shape[0]):
		class_label = np.argmax(y[i])
		y_pred_decoded[i][0] = class_label
	return y_pred_decoded


def softmax(Z):
	"""
	Softmax function
	:param Z: Input is output of regression function, NxK
	:return: The softmax(f) result of NxK
	"""
	exps = np.exp(Z)
	exps_sum = exps.sum(axis=1, keepdims=True)
	return exps / exps_sum


def regress(X, theta):
	"""
	Regression function
	:param X: input matrix of NxD
	:param theta: input parameter matrix containing bias b and weights w of DxK
	:return: Regression function output
	"""
	return X @ theta[1] + theta[0]


def multinoulli_log_likelihood(p, y):
	"""
	Multinoulli log likelihood
	:param p: output of softmax function of NxK
	:param y: actual output of NxK
	:return: the calculated multioulli log likelihood
	"""
	return - (np.sum(y * np.log(p))) / (y.shape[0])


def computeCost(X, y, theta, beta):
	"""
	Cost function for multinoulli log likelihood
	:param X: input matrix of NxD
	:param y: actual output of NxK
	:param theta: input parameter matrix containing bias b and weights w of DxK
	:param beta: meta paramter to control regularization
	:return: Loss/cost of the model
	"""
	return multinoulli_log_likelihood(softmax(regress(X, theta)), y) + (beta *
									np.sum(theta[1] ** 2) / 2)


def computeGrad(X, y, theta, beta):
	"""
	Compute gradient descent or derivatives of the parameters
	:param X: input matrix of NxD
	:param y: actual output of NxK
	:param theta: input parameter matrix containing bias b and weights w of DxK
	:param beta: meta paramter to control regularization
	:return: Derivatives of bias and weights
	"""
	# derivative w.r.t. to model output units (fy)
	dL_dfy = softmax(regress(X, theta)) - y
	# derivative w.r.t model weights b
	dL_db = (1 / y.shape[0]) * (np.sum(dL_dfy))
	# derivative w.r.t model weights w
	dL_dw = (1 / y.shape[0]) * (X.T @ dL_dfy) + (beta * theta[1])
	# nabla => the full gradient
	nabla = (dL_db, dL_dw)
	return nabla


def classify(X, theta):
	"""
	Predict for input using the given weights
	:param X: input matrix of NxD
	:param theta: input parameter matrix containing bias b and weights w of DxK
	:return: label outtput of Nx1
	"""
	# predicted output of NxK
	y_pred = softmax(regress(X, theta))
	# Decoded output of Nx1
	return decode_one_hot(y_pred)


##############################################################################
# Reading Data and transformations
K = 3 # Number of classes
path = os.getcwd() + '/data/iris_train.dat'
data = pd.read_csv(path, header=None, names=['X1','X2','X3','X4','Y'])
X = data[['X1','X2','X3','X4']].to_numpy()
Y = data[['Y']].to_numpy()
y = encode_one_hot(Y, K)

path = os.getcwd() + '/data/iris_test.dat'
data = pd.read_csv(path, header=None, names=['X1','X2','X3','X4','Y'])
X_test = data[['X1','X2','X3','X4']].to_numpy()
Y_test = data[['Y']].to_numpy()
y_test = encode_one_hot(Y_test, K)

w = np.zeros((X.shape[1], K))
b = np.array([0])
theta = (b, w)

############################################################################
print("=================================================================")
training_loss = []
validation_loss = []
mini_batches = create_mini_batch(X, y)
L = computeCost(X, y, theta, beta)
L_prev = L
print("-1 L = {0}".format(L))
halt = 0
i = 0
while(i < n_epoch and halt == 0):
	# Compute gradient through batch processing
	for batch in mini_batches:
		# Compute gradient for the mini-batch
		X_batch, y_batch = batch[0], batch[1]
		dL_db, dL_dw = computeGrad(X_batch, y_batch, theta, beta)
		b = theta[0]
		w = theta[1]

		# Update parameters
		b = b - (alpha * dL_db)
		w = w - (alpha * dL_dw)
		theta = (b, w)

	# Compute cost (training loss) and validation loss
	L = computeCost(X, y, theta, beta)
	training_loss.append(L)
	validation_cost = computeCost(X_test, y_test, theta, beta)
	validation_loss.append(validation_cost)

	# Halting mechanism
	if abs(L - L_prev) < eps:
		halt = 1
	L_prev = L

	print(" {0} L = {1}".format(i,L))
	i += 1

print("w = ",w)
print("b = ",b)

############################################################################
print("\n=================================================================")
print("Training Data")
# Checking accuracy
predictions = classify(X, theta)
err = 0.0
difference = Y - predictions
# Calculating accuracy [= accurate_predictions / total_predictions]
accuracy = len(difference[difference == 0]) / len(difference)
err = 1 - accuracy
print('Error = ' + str(round(err * 100, 3)) + '%')
print('Accuracy = ' + str(round(accuracy * 100, 3)) + '%')

############################################################################
print("\n=================================================================")
print("Test Data")
# Checking accuracy
predictions = classify(X_test, theta)
err = 0.0
difference = Y_test - predictions
# Calculating accuracy [= accurate_predictions / total_predictions]
accuracy = len(difference[difference == 0]) / len(difference)
err = 1 - accuracy
print('Error = ' + str(round(err * 100, 3)) + '%')
print('Accuracy = ' + str(round(accuracy * 100, 3)) + '%')

############################################################################
# Plotting Loss function for training loss and validation loss
print("\n=================================================================")
print("Plotting Loss function plot! Close it to continue...")
plt.plot(training_loss, label="Training Loss", color = 'r')
plt.plot(validation_loss, label="Validation Loss", color = 'g')
plt.ylabel("Cost of model")
plt.xlabel("epochs")
plt.legend(loc="best")
plt.suptitle("Plot for Loss as a function")
plt.title("Training and validation losses for the model")
# Saving plot to disk
output_path = os.getcwd() + '/out/'
# Create the folder if it doesn't exist
if not os.path.exists(output_path):
	os.makedirs(output_path)
plt.savefig(output_path + "q1c_loss_function_plot")
plt.show()

############################################################################
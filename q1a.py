"""
file: q1a.py
description: Problem 1a
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
n_epoch = 1000 # number of epochs (full passes through the dataset)
eps = 0.00001 # controls convergence criterion


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


def calculate_numerical_gradient(X, y, theta, beta):
	"""
	Calcuates numercial gradient using the secant approximation by checking
	change in loss with change with each parameter
	:param X: input matrix of NxD
	:param y: actual output of NxK
	:param theta: input parameter matrix containing bias b and weights w of DxK
	:param beta: meta paramter to control regularization
	:return: List of derivatives
	"""
	# flattenning the parameter array
	theta_flat = [theta[0], theta[1]]
	theta_flat = np.concatenate([x.ravel() for x in theta_flat])
	shape1 = theta[0].shape
	shape2 = theta[1].shape
	numerical_gradient_derivatives = []
	# for each parameter j in theta
	for j in range(len(theta_flat)):
		# Cost for J(theta + epsilon)==============================
		temp = np.copy(theta_flat)
		temp[j] += eps  # updating jth parameter
		# reshaping to original shape of theta
		new_b, new_w = np.split(temp, [np.prod(shape1)])
		new_theta = (new_b.reshape(shape1), new_w.reshape(shape2))
		L_right = computeCost(X, y, new_theta, beta)
		# Cost for J(theta - epsilon)==============================
		temp = np.copy(theta_flat)
		temp[j] -= eps  # updating jth parameter
		# reshaping to original shape of theta
		new_b, new_w = np.split(temp, [np.prod(shape1)])
		new_theta = (new_b.reshape(shape1), new_w.reshape(shape2))
		L_left = computeCost(X, y, new_theta, beta)
		# Approximation of derivative==============================
		derivative = (L_right - L_left) / (2 * eps)
		numerical_gradient_derivatives.append(derivative)
	return np.array(numerical_gradient_derivatives)


def check_gradient(X, y, theta, beta, nabla):
	"""
	Chekcing whether convergence actually happend by comparing the
	derivatives with the numerical calculated gradient derivatives
	:param X: input matrix of NxD
	:param y: actual output of NxK
	:param theta: input parameter matrix containing bias b and weights w of DxK
	:param beta: meta paramter to control regularization
	:param nabla: Latest derivatives of theta parameters
	:return: bool for every parameter, if it is correct (1x1, DxK)
	"""
	# Numerically calculating the derivatives
	numerical_gradient_derivatives = calculate_numerical_gradient(X, y,
																  theta,
																  beta)
	# Actual latest derivatives - flattened
	nabla = np.concatenate([x.ravel() for x in nabla])
	result = []
	# Comapring the derivatives
	for j in range(len(nabla)):
		# checking if difference is less than epsilon
		if abs(numerical_gradient_derivatives[j] - nabla[j]) < 1e-4:
			result.append('CORRECT')
		else:
			result.append('INCORRECT')
	# reshaping the results to original shape of theta
	shape1 = theta[0].shape
	shape2 = theta[1].shape
	dL_b_check, dL_w_check = np.split(result, [np.prod(shape1)])
	result = (dL_b_check.reshape(shape1), dL_w_check.reshape(shape2))
	return result


##############################################################################
# Reading Data and transformations
K = 2 # Number of classes
path = os.getcwd() + '/data/xor.dat'
data = pd.read_csv(path, header=None, names=['X1','X2','Y'])
X = data[['X1','X2']].to_numpy()
Y = data[['Y']].to_numpy()
y = encode_one_hot(Y, K)

# Initializing parameters in theta
np.random.seed(13) # Seeding randomness of weight values
w = np.random.rand(X.shape[1], K)
# w = np.zeros((X.shape[1], K))
b = np.array([0])
theta = (b, w)

# print(regress(X, theta))
# print(softmax(regress(X, theta)))
# print(multinoulli_log_likelihood(softmax(regress(X, theta)), y))

############################################################################
print("=================================================================")
L = computeCost(X, y, theta, beta)
L_prev = L
halt = 0
print("-1 L = {0}".format(L))
i = 0
while(i < n_epoch and halt == 0):
	# Compute gradient
	dL_db, dL_dw = computeGrad(X, y, theta, beta)
	b = theta[0]
	w = theta[1]

	# Update parameters
	b = b - (alpha * dL_db)
	w = w - (alpha * dL_dw)
	theta = (b, w)

	# Compute cost
	L = computeCost(X, y, theta, beta)

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
print("Gradient Convergence Check")
# Gradient convergence check
check = check_gradient(X, y, theta, beta, [np.array([dL_db]), dL_dw])
print(theta)
print(check)


############################################################################
print("\n=================================================================")
print("Misclassification Errors")
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
# Creating the input range for the decision boundary
X1_min, X1_max = np.min(X[:, 0]), np.max(X[:, 0])
X2_min, X2_max = np.min(X[:, 1]), np.max(X[:, 1])
xx1, xx2 = np.meshgrid(np.linspace(X1_min, X1_max), np.linspace(X2_min, X2_max))
x_bound = np.c_[xx1.ravel(), xx2.ravel()]
y_pred = decode_one_hot(regress(x_bound, theta))
y_pred = y_pred.reshape(xx1.shape)

print("\n=================================================================")
print("Plotting scatter plot! Close it to continue...")
# Plotting decision boundary on the scatter plot of data
f, ax = plt.subplots(figsize=(8, 6))
contour = ax.contour(xx1, xx2, y_pred, levels=[.5], cmap="Greys", vmin=0,
					 vmax=.6)
scatter = ax.scatter(X[:, 0], X[:, 1], s=50, c=np.squeeze(Y),
           cmap="viridis",
           vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)
ax.set(aspect="equal",
       xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
       xlabel="$X_1$", ylabel="$X_2$")
# Creating Labels for the Handles
handles, _ = contour.legend_elements()
handles2, _ = scatter.legend_elements()
labels = ["Decision boundary", "Class 0", "Class 1", "Class 2"]
plt.legend(handles + handles2, labels,loc="lower right")
plt.suptitle("Scatter Plot")
plt.title("Multiclass Regression on xor.dat")
# Saving plot to disk
output_path = os.getcwd() + '/out/'
# Create the folder if it doesn't exist
if not os.path.exists(output_path):
	os.makedirs(output_path)
plt.savefig(output_path + "q1a_decision_boundary_plot")
plt.show()

############################################################################

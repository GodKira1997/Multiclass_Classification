"""
file: q2.py
description: Problem 2
language: python3
author: Anurag Kallurwar, ak6491@rit.edu
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# GLOBAL
TARGET = " is spam" # Target or output of the data


def compute_for_discrete(data, feature, feature_value):
    """
    Compute aposteriori probability for the discrete feature
    :param data: Input Dataset either positive or negative
    :param feature: feature Xi
    :param feature_value: Categorical value
    :return: value of P(Xi = xj | Y = yk)
    """
    # P(feature Xi = feature_value | y (TARGET) = True (if positive data) or
    # False (if negative data))
    return data[data[feature].isin([feature_value])].shape[0] / data.shape[0]


def compute_for_continuous(data, feature):
    """
    Compute mean and standard deviation for the continuous feature
    :param data: Input Dataset either positive or negative
    :param feature: feature Xi
    :return: (mean, standard deviation)
    """
    # convert data in continuous feature's column to numpy array
    series = data[feature].to_numpy()
    # calculate mean and standard deviation for the feature
    return (np.mean(series), np.std(series))


def compute_maximum_likelihood_parameters(data):
    """
    Compute the maximum likelihood parameters for the data
    :param data: Input Dataset
    :return: dictionary of maximum likelihood parameters
    """
    # Fetching the feature names from the data
    features = data.columns.tolist()
    features.remove(TARGET)
    # Initializations
    maximum_likelihood_parameters = dict()
    # Splitting data into classes
    positives = data[data[TARGET].isin([True])]
    negatives = data[data[TARGET].isin([False])]
    # P(y = True)
    maximum_likelihood_parameters[(True)] = positives.shape[0] / data.shape[0]
    # P(y = False)
    maximum_likelihood_parameters[(False)] = negatives.shape[0] / data.shape[0]
    # For every feature of the dataset
    for feature in features:
        # calculate the aposteriori probabilitites
        if data[feature].dtype == bool: # For discrete features
            # P(Xi = True | Y = True)
            maximum_likelihood_parameters[((feature, True), True)] = \
                compute_for_discrete(positives, feature, True)
            # P(Xi = True | Y = False)
            maximum_likelihood_parameters[((feature, True), False)] = \
                compute_for_discrete(negatives, feature, True)
            # P(Xi = False | Y = True)
            maximum_likelihood_parameters[((feature, False), True)] = \
                compute_for_discrete(positives, feature, False)
            # P(Xi = False | Y = False)
            maximum_likelihood_parameters[((feature, False), False)] = \
                compute_for_discrete(negatives, feature, False)
        else: # For continuous features
            # mean,std for P(Xi | Y = True)
            maximum_likelihood_parameters[(feature, True)] = \
                compute_for_continuous(positives, feature)
            # mean,std for P(Xi = True | Y = True)
            maximum_likelihood_parameters[(feature, False)] = \
                compute_for_continuous(negatives, feature)
    return maximum_likelihood_parameters


def print_maximum_likelihood_parameters(maximum_likelihood_parameters, data):
    """
    Print the dictionary of maximum likelihood parameters
    :param maximum_likelihood_parameters: dictionary of maximum likelihood
    parameters
    :param data: Dataset
    :return: None
    """
    features = data.columns.tolist()
    features.remove(TARGET)
    # Probabilities for class labels
    print("P('"  + TARGET + "' = " + str(True) + ") = " + str(round(
        maximum_likelihood_parameters[True], 3)))
    print("P('" + TARGET + "' = " + str(False) + ") = " + str(round(
        maximum_likelihood_parameters[False], 3)))
    # Probabilitites for each feature
    for feature in features:
        if data[feature].dtype == bool: # Discrete feature
            print("P('" + feature + "' = " + str(True) + " | '" + TARGET +
                  "' = " + str(True) + ") = " +
                  str(round(maximum_likelihood_parameters[((feature, True),
                                                           True)], 3)))
            print("P('" + feature + "' = " + str(True) + " | '" + TARGET +
                  "' = " + str(False) + ") = " +
                  str(round(maximum_likelihood_parameters[((feature, True),
                                                           False)], 3)))
            print("P('" + feature + "' = " + str(False) + " | '" + TARGET +
                  "' = " + str(True) + ") = " +
                  str(round(maximum_likelihood_parameters[((feature, False),
                                                           True)], 3)))
            print("P('" + feature + "' = " + str(False) + " | '" + TARGET +
                  "' = " + str(False) + ") = " +
                  str(round(maximum_likelihood_parameters[((feature, False),
                                                           False)], 3)))
        else: # Continuous feature
            print("P('" + feature + "' | '" + TARGET + "' = " + str(True) +
                  ") = (" +
                  str(round(maximum_likelihood_parameters[(feature, True)][
                                0], 3)) + ", " + str(round(
                maximum_likelihood_parameters[(feature, True)][1], 3)) + ")")
            print("P('" + feature + "' | '" + TARGET + "' = " + str(False) +
                  ") = (" +
                  str(round(maximum_likelihood_parameters[(feature, False)][
                                0], 3)) + ", " + str(round(
                maximum_likelihood_parameters[(feature, False)][1], 3)) + ")")


def gaussian_pdf(x, mean, std):
    """
    Calculate the probability density function for the input X with the given
    mean and std
    :param x: input x
    :param mean: mean of data
    :param std: standard deviation of data
    :return:
    """
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean) ** 2) /
                                                     (2 * std ** 2))


def classify(X, maximum_likelihood_parameters):
    """
    Predict for input X using the maximum likelihood parameters
    :param X: Input X, NxD (D can be less than actual D (subset))
    :param maximum_likelihood_parameters: dictionary of maximum likelihood
    parameters
    :return: Predicted classes, Nx1
    """
    features = X.columns.tolist()
    y_pred = []
    # Every row in input X preedict class label
    for index, row in X.iterrows():
        # Predict for current row vector using values and dictionary of
        # maximum likelihood parameters
        class_value = False
        # Initializing the aposteriori probabilities for class labels
        aposteriori_true = maximum_likelihood_parameters[True]
        aposteriori_false = maximum_likelihood_parameters[False]
        # Calculating aposterioris for current row vectors values
        for feature in features:
            if X[feature].dtype == bool: # discrete feature
                aposteriori_true *= \
                    maximum_likelihood_parameters[((feature, row[feature]),
                                                   True)]
                aposteriori_false *= \
                    maximum_likelihood_parameters[((feature, row[feature]),
                                                   False)]
            else: # continous feature
                mean, std = maximum_likelihood_parameters[(feature, True)]
                aposteriori_true *= gaussian_pdf([row[feature]], mean, std)
                mean, std = maximum_likelihood_parameters[(feature, False)]
                aposteriori_false *= gaussian_pdf([row[feature]], mean, std)
        # print(aposteriori_true, aposteriori_false)
        # get the maximum aposteriori probability (MAP)
        if aposteriori_true > aposteriori_false:
            class_value = True
        y_pred.append(class_value)
    # print(y_pred)
    return np.array(y_pred)


def predict(data_b, features, maximum_likelihood_parameters):
    """
    Predict for dataset b based on given subset of features
    :param data_b: Input dataset
    :param features: features to be used
    :return: None
    """
    # Keeping only the given features
    data_b = data_b[features]

    ###########################################################################
    # Predicting class labels for testing dataset
    features.remove(TARGET)
    X = data_b[features]
    y = data_b[TARGET]
    y = np.array(y.values)

    ###########################################################################
    print("\n=================================================================")
    print("Misclassification Errors")
    # Classify into labels
    predictions = classify(X, maximum_likelihood_parameters)
    err = 0.0
    correct = np.sum(predictions == y)
    print
    # Calculating accuracy [= accurate_predictions / total_predictions]
    accuracy = correct / len(predictions)
    err = 1 - accuracy
    print('Error = ' + str(round(err * 100, 3)) + '%')
    print('Accuracy = ' + str(round(accuracy * 100, 3)) + '%')


############################################################################
# Reading Data and transformations
path_a = os.getcwd() + '/data/q3.csv'
data_a = pd.read_csv(path_a)
path_b = os.getcwd() + '/data/q3b.csv'
data_b = pd.read_csv(path_b)

############################################################################
# Training based on dataset data_a
print("\n=================================================================")
print("Maximum Likelihood Parameters")
# Calculating maximum likelihood parameters
maximum_likelihood_parameters = compute_maximum_likelihood_parameters(
    data_a)
print_maximum_likelihood_parameters(maximum_likelihood_parameters, data_a)

############################################################################
print("\n=================================================================")
print("Naives Bayes's Classifier based on all features")
features = data_a.columns.tolist()
print(features)
predict(data_b, features, maximum_likelihood_parameters)

############################################################################
print("\n=================================================================")
print("Naives Bayes's Classifier based on selected features")
features_subset = [' has emoji', ' from .com', ' has my name',
                  ' # sentences', ' # words', ' is spam']
print(features_subset)
predict(data_b, features_subset, maximum_likelihood_parameters)

############################################################################

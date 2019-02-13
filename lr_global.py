import numpy as np
from numpy import linalg as LA
from math import ceil

np.set_printoptions(suppress=True)

def compute_rmse(targets, expected):
    return np.sqrt(((targets - expected) ** 2).mean())

def linear_regression(features, targets):
    return LA.inv(features.T @ features) @ features.T @ targets

def separate_data(data):
    targets = data[:, 2:]
    features = data[:, :2]
    return (targets, features)

def standardize(features, mean=None, std=None):
    features = (features - mean) / std
    ones = np.ones((features.shape[0], 1))
    features = np.append(ones, features, axis=1)
    return features

def main():
    data = np.genfromtxt('./x06Simple.csv', delimiter=',', dtype="uint16", skip_header=1, usecols=(1,2,3))

    np.random.seed(0)
    np.random.shuffle(data)

    range = ceil(len(data) * 2/3)

    training = data[0:range]
    testing = data[range:]

    training_targets, training_features = separate_data(training)
    testing_targets, testing_features = separate_data(testing)

    mean = np.mean(training_features, axis=0)
    std = np.std(training_features, axis=0, ddof=1)

    training_features = standardize(training_features, mean, std)

    theta = linear_regression(training_features, training_targets)

    testing_features = standardize(testing_features, mean, std)
    expected = testing_features @ theta

    rmse = compute_rmse(testing_targets, expected)

    print("rmse", rmse)
if __name__ == "__main__":
    main()

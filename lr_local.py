import numpy as np
from numpy import linalg as LA
from math import ceil

np.set_printoptions(suppress=True)

def compute_rmse(targets, expected):
    return np.sqrt(((targets - expected) ** 2).mean())

def compute_distance(a, b, k=1):
    distances = []
    for item in b:
        #test = np.sqrt(np.sum((a - item) ** 2))
        #test = -(a - item) ** 2
        #distances.append(np.sum(test))

        distances.append( -(LA.norm(a - item) / k) )

    w = np.diag(np.exp(distances))

    return w

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
    #training_features = np.array([ [1], [3], [5], [6] ])
    #training_targets = np.array([ [6], [9], [17], [12] ])
    #testing_features = np.array([ [2], [4]])
    #testing_targets = np.array([ [1], [5] ])

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
    testing_features = standardize(testing_features, mean, std)

    expectations = []

    for test in testing_features:
        w = compute_distance(test, training_features)
        theta = LA.inv(training_features.T @ w @ training_features) @ training_features.T @ w @ training_targets
        expectation = test @ theta
        expectations.append(expectation)

    expectations = np.array(expectations)

    rmse = compute_rmse(testing_targets, expectations)
    print("Root mean squared error:", rmse)

if __name__ == "__main__":
    main()

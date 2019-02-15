import numpy as np
from numpy import linalg as LA
from math import ceil
from standardization import standardize, separate_data

np.set_printoptions(suppress=True)

def compute_rmse(targets, expected):
    return np.sqrt(((targets - expected) ** 2).mean())

def compute_distance(a, b, k=1):
    distances = []
    for item in b:
        distance = -(LA.norm((a - item), ord=1) / k ** 2)
        distances.append(distance)

    w = np.diag(np.exp(distances))
    return w

def main():
    data = np.genfromtxt('./x06Simple.csv', delimiter=',', dtype="uint16", skip_header=1, usecols=(1,2,3))

    np.random.seed(0)
    np.random.shuffle(data)

    range = ceil(len(data) * 2/3)

    training = data[0:range]
    testing = data[range:]

    training_targets, training_features = separate_data(training)
    testing_targets, testing_features = separate_data(testing)

    #training_features = np.array([ [1], [3], [5], [6] ])
    #training_targets = np.array([ [6], [9], [17], [12] ])
    #testing_features = np.array([ [2], [4]])
    #testing_targets = np.array([ [1], [5] ])

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

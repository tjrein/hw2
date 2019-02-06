import numpy as np
from numpy import linalg as LA
from math import ceil

np.set_printoptions(suppress=True)

def linear_regression(data):
    targets = data[:, 3:]
    features = data[:, :3]

    #test equation
    #targets = np.array([ [6], [1], [9], [17], [12]])
    #features = np.array([ [-1.1574], [-0.6751], [-0.1929], [0.7716], [1.2538] ])
    #ones = np.ones((5, 1))
    #features = np.append(ones, features, axis=1)


    theta = LA.inv(features.T @ features) @ features.T @ targets
    expected = features @ theta

    return expected

def standardize(data):
    targets = data[:, 2:]
    test = data[:, :2]

    mean = np.mean(test, axis=0)
    std = np.std(test, axis=0, ddof=1)

    ones = np.ones((30, 1))

    standardized_data = (test - mean) / std
    standardized_data = np.append(ones, standardized_data, axis=1)
    standardized_data = np.append(standardized_data, targets, axis=1)

    return standardized_data

def main():
    data = np.genfromtxt('./x06Simple.csv', delimiter=',', dtype="uint16", skip_header=1, usecols=(1,2,3))

    np.random.seed(0)
    np.random.shuffle(data)

    range = ceil(len(data) * 2/3)

    training = data[0:range]
    test = data[range:]

    training = standardize(training)

    linear_regression(training)

if __name__ == "__main__":
    main()

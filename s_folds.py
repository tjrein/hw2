import numpy as np
import sys
from numpy import linalg as LA

def compute_se(targets, expected):
    return (targets - expected) ** 2

def compute_rmse(se):
    return np.sqrt(se.mean())

def separate_data(data):
    targets = data[:, 2:]
    features = data[:, :2]
    return (targets, features)

def standardize(features, mean=None, std=None):
    features = (features - mean) / std
    ones = np.ones((features.shape[0], 1))
    features = np.append(ones, features, axis=1)
    return features

def perform_s_folds(s, data):
    se = []
    for i in range(0, len(data)):
        testing =  data[i]
        training = [x for j, x in enumerate(data) if j != i]
        training = np.vstack(training)

        training_targets, training_features = separate_data(training)
        testing_targets, testing_features = separate_data(testing)

        mean = np.mean(training_features, axis=0)
        std = np.std(training_features, axis=0, ddof=1)

        training_features = standardize(training_features, mean, std)
        testing_features = standardize(testing_features, mean, std)

        theta = LA.inv(training_features.T @ training_features) @ training_features.T @ training_targets
        expected = testing_features @ theta
        error = compute_se(testing_targets, expected)
        se.append(error)

    se = np.vstack(se)
    rmse = compute_rmse(se)
    return rmse

def main():
    args = sys.argv
    data = np.genfromtxt('./x06Simple.csv', delimiter=',', dtype="uint16", skip_header=1, usecols=(1,2,3))
    s = 3

    if len(args) > 1:
        if args[1].isdigit():
            s = int(args[1])
            if s == 1:
                return print("S-Folds must be larger than 1")
        elif args[1].lower() == 'n':
            s = len(data)
        else:
            return

    all_rmse = []
    for i in range(0, 20):
        np.random.seed(i)
        np.random.shuffle(data)
        split_data = np.array_split(data, s)
        rmse = perform_s_folds(s, split_data)
        all_rmse.append(rmse)

    rmse_mean = np.mean(all_rmse, axis=0)
    rmse_std = np.std(all_rmse, axis=0, ddof=1)

    print("rmse_mean", rmse_mean)
    print("rmse_std", rmse_std)

    return

if __name__ == '__main__':
    main()

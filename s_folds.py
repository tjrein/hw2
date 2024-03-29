import numpy as np
import sys
from numpy import linalg as LA
from data_operations import standardize, isolate_sets

def compute_se(y, expected):
    return (y - expected) ** 2

def compute_rmse(se):
    return np.sqrt(se.mean())

def perform_s_folds(s, data):
    se = []
    for i in range(0, len(data)):
        testing =  data[i]
        training = [x for j, x in enumerate(data) if j != i]
        training = np.vstack(training)

        train_x, train_y, test_x, test_y = isolate_sets(training, testing)

        theta = LA.inv(train_x.T @ train_x) @ train_x.T @ train_y
        expected = test_x @ theta
        error = compute_se(test_y, expected)
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

    print(f"Average RMSE over {s} folds:", rmse_mean)
    print(f"Standard deviation over {s} folds:", rmse_std)

    return

if __name__ == '__main__':
    main()

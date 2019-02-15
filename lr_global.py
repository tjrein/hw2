import numpy as np
from numpy import linalg as LA
from math import ceil
from standardization import standardize, separate_data, handle_data, compute_rmse

def main():
    data = np.genfromtxt('./x06Simple.csv', delimiter=',', dtype="uint16", skip_header=1, usecols=(1,2,3))
    train_x, train_y, test_x, test_y = handle_data(data)
    theta = LA.inv(train_x.T @ train_x) @ train_x.T @ train_y
    expected = test_x @ theta
    rmse = compute_rmse(test_y, expected)

    print("Theta:\n", theta)
    print("Root mean squared error:", rmse)

if __name__ == "__main__":
    main()

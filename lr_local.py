import numpy as np
from numpy import linalg as LA
from data_operations import handle_data, compute_rmse

def compute_model(a, b, k=1):
    distances = []

    for item in b:
        distance = -(LA.norm((a - item), ord=1) / k ** 2)
        distances.append(distance)

    w = np.diag(np.exp(distances))
    return w

def main():
    data = np.genfromtxt('./x06Simple.csv', delimiter=',', dtype="uint16", skip_header=1, usecols=(1,2,3))
    train_x, train_y, test_x, test_y = handle_data(data)
    expectations = []

    for obs in test_x:
        w = compute_model(obs, train_x)
        theta = LA.inv(train_x.T @ w @ train_x) @ train_x.T @ w @ train_y
        expectation = obs @ theta
        expectations.append(expectation)

    expectations = np.array(expectations)
    rmse = compute_rmse(test_y, expectations)

    print("Root mean squared error:", rmse)

if __name__ == "__main__":
    main()

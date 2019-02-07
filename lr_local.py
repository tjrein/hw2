import numpy as np
from numpy import linalg as LA

np.set_printoptions(suppress=True)

def compute_distance(a, b):
    distances = []
    for item in b:
        distances.append( -(LA.norm(a - item) / 1) )

    w = np.diag(np.exp(distances))

    return w

def main():
    x_train = np.array([ [1, 1], [1,3], [1, 5], [1, 6] ])
    y_train = np.array([ [6], [9], [17], [12] ])
    x_test = np.array([ [1, 2], [1, 4]])
    y_test = np.array([ [1, 5] ])


    w = compute_distance(x_test[0,:], x_train)
    theta = LA.inv(x_train.T @ w @ x_train) @ x_train.T @ w @ y_train
    expectation = x_test[0,:] @ theta

    expectations = []

    for test in x_test:
        w = compute_distance(test, x_train)
        theta = LA.inv(x_train.T @ w @ x_train) @ x_train.T @ w @ y_train
        expectation = test @ theta
        expectations.append(expectation)

    print("expectations", np.array(expectations))


if __name__ == "__main__":
    main()

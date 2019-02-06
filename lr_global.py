import numpy as np
from math import ceil

np.set_printoptions(suppress=True)

def standardize(data):

    print("data", data[:, :2])
    targets = data[:, 2: ]

    test = data[:, :2]

    mean = np.mean(test, axis=0)
    std = np.std(test, axis=0, ddof=1)

    print("mean", mean)
    print("std", std)

    standardized_data = (test - mean) / std

    #print("standardized", standardized_data)
    #targets.reshape((30, 1)
    #print("targets shape", targets.shape)
    standardized_data = np.append(standardized_data, targets, axis=1)
    print("standardized_dat\n", standardized_data)
    return
    return standardized_data

def main():
    data = np.genfromtxt('./x06Simple.csv', delimiter=',', dtype="uint16", skip_header=1, usecols=(1,2,3))

    np.random.seed(0)
    np.random.shuffle(data)

    print("len", len(data))

    range = ceil(len(data) * 2/3)


    training = data[0:range]
    test = data[range:]

    print("training", training)
    print("\n")

    training = standardize(training)


if __name__ == "__main__":
    main()

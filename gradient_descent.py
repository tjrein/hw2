import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from math import ceil
from standardization import standardize, separate_data

def compute_rmse(targets, expected):
    return np.sqrt(((targets - expected) ** 2).mean())

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
    testing_features = standardize(testing_features, mean, std)

    random_thetas = 2 * np.random.random_sample((3, 1)) - 1
    learning_rate = 0.01

    change_in_rmse = 1
    iterations = 0

    thetas = random_thetas
    initial_expected_values = training_features @ thetas

    rmse = compute_rmse(initial_expected_values, training_targets)

    plot = ([iterations + 1], [rmse], [compute_rmse(testing_features @ thetas, testing_targets)])

    while iterations <= 1000000:
        gradient = training_features.T @ (training_features @ thetas - training_targets)
        thetas = thetas - (learning_rate * gradient / len(training_features))

        expected = training_features @ thetas
        expected_testing = testing_features @ thetas

        new_rmse = compute_rmse(training_targets, expected)
        testing_rmse = compute_rmse(testing_targets, expected_testing)

        percent_change = np.abs((new_rmse - rmse) / rmse * 100)

        if percent_change <= 2 ** -23:
            break

        iterations += 1
        rmse = new_rmse

        plot[0].append(iterations)
        plot[1].append(rmse)
        plot[2].append(testing_rmse)

    final_expected = testing_features @ thetas
    final_rmse = compute_rmse(testing_targets, final_expected)

    print("Thetas:\n", thetas)
    print("Root mean square error:", final_rmse)

    plt.plot(plot[0], plot[1], label="training")
    plt.plot(plot[0], plot[2], label="testing")
    plt.xlabel("Iterations")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()

    return

if __name__ == '__main__':
    main()

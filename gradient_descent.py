import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from data_operations import handle_data, compute_rmse

def plot_gr(plot):
    plt.plot(plot[0], plot[1], label="train")
    plt.plot(plot[0], plot[2], label="test")
    plt.xlabel("Iterations")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig("gradient_descent", bbox_inches="tight")
    plt.show()

def main():
    data = np.genfromtxt('./x06Simple.csv', delimiter=',', dtype="uint16", skip_header=1, usecols=(1,2,3))
    train_x, train_y, test_x, test_y = handle_data(data)
    random_thetas = 2 * np.random.random_sample((3, 1)) - 1
    learning_rate = 0.01
    iterations = 0

    thetas = random_thetas

    rmse = compute_rmse(train_x @ thetas, train_y)
    test_rmse = compute_rmse(test_x @ thetas, test_y)

    plot = ([iterations + 1], [rmse], [test_rmse])

    while iterations <= 1000000:
        gradient = train_x.T @ (train_x @ thetas - train_y)
        thetas = thetas - (learning_rate * gradient / len(train_x))

        expected = train_x @ thetas
        expected_test = test_x @ thetas

        new_rmse = compute_rmse(train_y, expected)
        test_rmse = compute_rmse(test_y, expected_test)

        percent_change = np.abs((new_rmse - rmse) / rmse * 100)

        if percent_change <= 2 ** -23:
            break

        iterations += 1
        rmse = new_rmse

        plot[0].append(iterations)
        plot[1].append(rmse)
        plot[2].append(test_rmse)

    final_expected = test_x @ thetas
    final_rmse = compute_rmse(test_y, final_expected)

    print("Thetas:\n", thetas)
    print("Root mean square error:", final_rmse)

    plot_gr(plot)

    return

if __name__ == '__main__':
    main()

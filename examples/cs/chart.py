import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_random_data():
    data = np.random.rand(10, 2)
    df = pd.DataFrame(data, columns=["x", "y"])

    plt.scatter(df["x"], df["y"])
    plt.title("Random Data Scatter Plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()


if __name__ == "__main__":
    plot_random_data()

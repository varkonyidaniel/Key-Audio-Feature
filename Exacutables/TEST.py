from sklearn.datasets import load_iris
from sklearn import tree
from matplotlib import pyplot as plt
import numpy as np


if __name__ == "__main__":
    a = np.array([1,2,3])

    for i in range(5):
        if i == 0:
            c = a
        else:
            c = np.vstack((c, a))

    print(c)

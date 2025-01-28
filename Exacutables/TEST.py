import numpy as np
from Outputs import data_reader as dr
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz





if __name__ == "__main__":

    alma = np.array([5,4,3,2,1])
    x = 2
    print(alma[:x])

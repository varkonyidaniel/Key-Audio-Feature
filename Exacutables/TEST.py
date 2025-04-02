import numpy as np
from Outputs import data_reader as dr
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz


if __name__ == "__main__":

    tr_idxs = '22'
    ts_idxs = '30,37'

    print(tr_idxs.replace(',','_'))
    print(ts_idxs.replace(',','_'))

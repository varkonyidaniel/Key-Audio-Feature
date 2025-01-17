from sklearn.cluster import KMeans
import numpy as np
import subprocess
import random
from operator import itemgetter
from Enum.enum_types import Regression_method as rm
from multiprocessing import Process

def func(idx:int):
    print(f"thread: {idx} func starting")
    for i in range(10000000):
        if i%1000000==0:
            print(f"thread:{idx}, step:{i}")
    print(f"thread: {idx} func finishing")


if __name__ == "__main__":

    proc = []
    for _ in range(3):
        p = Process(target=func,args=(_,))
        p.start()
        proc.append(p)
    for p in proc:
        p.join()

    for i in range(10):
        print(f"main function: {i}")

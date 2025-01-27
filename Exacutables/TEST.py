import numpy as np
from Outputs import data_reader as dr

if __name__ == "__main__":


    f = "26_2020_05_26_09_30_26_timestamps.h5"
    s = dr.read_h5_file("../DATA/FE/",f"{f}","timelabels")

    print(s)



#

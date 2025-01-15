import matplotlib.pyplot as plt
import numpy as np


def drawHeatMap(data, xvalues, yvalues, xlabel: str = '', ylabel: str = '', title: str = '',
                save_fig:bool=False, dpi:int=300, filename:str=""):
        plt.figure(figsize=(12, 6))
        #plt.imshow(data, cmap="hot_r",extent=[0, 8.2, 4000, 0], aspect='auto')
        plt.imshow(data, cmap="hot_r", aspect='auto')
        plt.gca().invert_yaxis()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(filename )
        plt.colorbar()

        if save_fig:
            plt.savefig(f'hm_{filename}.png',dpi=dpi)
        plt.show()


def drawTimeSeries(data: np.ndarray, xlabel, ylabel, text:str='', line_y:bool=False, save_fig:bool=False,
                   xvalues:np.ndarray=None, dpi:int=300):
    #plt.figure(figsize=(16, 8))
    #if line_y:
    #    plt.axhline(y=0.0, color='r', linestyle='-')
    #plt.plot(xvalues,data)
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(text)
    #if save_fig:
    #    plt.savefig(f'ts_{text}.png',dpi=dpi)
    plt.show()

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    __loc__ = os.path.realpath(os.path.join(os.getcwd(),os.path.dirname(__file__)))
    filepath = os.path.join(__loc__,'profiling.xlsx')

    df = pd.read_excel(filepath)

    plt.plot(df['n'],df['time'])
    plt.xlabel('Number of elements in line')
    plt.ylabel('Time to iterate')
    plt.show()
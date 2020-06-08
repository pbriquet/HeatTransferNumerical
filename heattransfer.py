import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    Lx, Ly = [1.0,1.0]
    nx, ny = [10,10]
    x = np.linspace(0.0,Lx,num=nx)
    y = np.linspace(0.0,Ly,num=ny)

    xx,yy = np.meshgrid(x,y,indexing='ij')
    T = np.ones(shape=xx.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(xx,y,T,cmap='jet')
    plt.show()


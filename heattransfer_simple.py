import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    Lx, Ly = [10.0e-2,20.0e-2]
    nx, ny = [10,20]
    x = np.linspace(0.0,Lx,num=nx)
    y = np.linspace(0.0,Ly,num=ny)

    xx,yy = np.meshgrid(x,y,indexing='ij')
    
    K = 100.0
    
    rho = 7.8e3
    Cp = 400.0
    alfa = K/rho/Cp

    dx,dy = [Lx/nx,Ly/ny]

    beta = 1.0
    dt = beta*min(dx,dy)**2/4.0
    T = np.ones(shape=xx.shape)
    Tn = np.ones(shape=xx.shape)
    
    print(dt)
    T[0,:] = 0.0
    T[-1,:] = 0.0
    T[:,0] = 0.0
    T[:,-1] = 0.0

    p = (slice(1,-1),slice(1,-1))
    e = (slice(2,None),slice(1,-1))
    w = (slice(0,-2),slice(1,-1))
    n = (slice(1,-1),slice(2,None))
    s = (slice(1,-1),slice(0,-2))
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(xx,y,T,cmap='jet')
    plt.show()

    tmax = 1e6*dt
    t = 0.0
    while(t < tmax):
        Tn[p] = T[p] + dt*alfa*(T[e] - 2.0*T[p] + T[w])/dx**2 + dt*alfa*(T[n] - 2.0*T[p] + T[s])/dy**2
        Tn[0,:] = 0.0
        Tn[-1,:] = 0.0
        Tn[:,0] = 0.0
        Tn[:,-1] = 0.0
        T[:,:] = Tn[:,:]
        t += dt
    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(xx,y,T,cmap='jet')
    plt.show()

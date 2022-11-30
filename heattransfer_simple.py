import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import line_profiler

class MeshGrid:
    def __init__(self,nx):
        self.nx = nx
        self.dim = len(nx)




if __name__ == "__main__":

    N = np.arange(11,20)

    for n in N:
        Lx, Ly = [10.0e-2,20.0e-2]
        nx, ny = [n*10,n*10]
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


        tmax = 1e4*dt
        Cx = dt*alfa/dx**2 
        Cy = dt*alfa/dy**2 

        def iterate():
            t = 0.0
            while(t < tmax):
                Tn[p] = T[p] + (T[e] - 2.0*T[p] + T[w])*Cx + (T[n] - 2.0*T[p] + T[s])*Cy
                Tn[0,:] = 0.0
                Tn[-1,:] = 0.0
                Tn[:,0] = 0.0
                Tn[:,-1] = 0.0
                T[:,:] = Tn[:,:]
                t += dt
        
        lp = line_profiler.LineProfiler()
        wrap = lp(iterate)
        wrap()
        lp.print_stats()
        #iterate()
        '''
        print(t)
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.plot_surface(xx,y,T,cmap='jet')
        plt.show()

        '''
        
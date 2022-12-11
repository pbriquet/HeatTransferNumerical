import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import line_profiler
import big_o

class MeshGrid2D:
    def __init__(self,Lx,Ly,nx,ny):
        self.nx = nx





if __name__ == "__main__":

    def func(N):
        Lx, Ly = [10.0e-2,20.0e-2]
        nx, ny = [N,N]
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
        
        T[0,:] = 0.0
        T[-1,:] = 0.0
        T[:,0] = 0.0
        T[:,-1] = 0.0

        p = (slice(1,-1),slice(1,-1))
        e = (slice(2,None),slice(1,-1))
        w = (slice(0,-2),slice(1,-1))
        n = (slice(1,-1),slice(2,None))
        s = (slice(1,-1),slice(0,-2))
        
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.plot_surface(xx,y,T,cmap='jet')
        '''

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
        print(N)
        iterate()
        '''
        lp = line_profiler.LineProfiler()
        wrap = lp(iterate)
        wrap()
        lp.print_stats()
        #iterate()
        '''
        '''
        print(t)
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.plot_surface(xx,y,T,cmap='jet')
        plt.show()

        '''
    def find_max(x):
        """Find the maximum element in a list of positive integers."""
        max_ = 0
        for el in x:
            if el > max_:
                max_ = el
        return max_
    positive_int_generator = lambda n: big_o.datagen.integers(n, 0 , 10000)
    best, others = big_o.big_o(func, big_o.datagen.n_, n_repeats=1, min_n=2, max_n=500)
    print(best)
    print(others)
        
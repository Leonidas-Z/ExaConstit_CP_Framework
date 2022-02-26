import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from deap import tools

class ExaPlots:          

    def ObjFun3D(ref_points, pop_fit, best_idx):

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")

        # Coordinate origin
        ax.scatter(0, 0, 0, c="k", marker="+", s=100)
        ax = fig.add_subplot(111,projection='3d') 

        '''
        # reference points
        # Parameters
        NOBJ = 3
        P = [2, 1]
        SCALES = [1, 0.5]

        # Create, combine and removed duplicates
        ref_points = [tools.uniform_reference_points(NOBJ, p, s) for p, s in zip(P, SCALES)]
        ref_points = numpy.concatenate(ref_points, axis=0)
        _, uniques = numpy.unique(ref_points, axis=0, return_index=True)
        ref_points = ref_points[uniques]
        ##
        for subset, p, s in zip(ref_points, P, SCALES):
        ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2], marker="o", s=48, label="p = {}, scale = {}".format(p, s))
        '''

        # Plot best_solution
        if best_idx:
            ax.scatter(pop_fit[best_idx,0], pop_fit[best_idx,1], pop_fit[best_idx,2], marker='o', linewidths=1, facecolors='none', edgecolors='black', s=60) 

        # Plot ref_points
        ax.scatter(ref_points[:,0],ref_points[:,1],ref_points[:,2], marker="o")

        # Plot last population fitness
        ax.scatter(pop_fit[:,0], pop_fit[:,1], pop_fit[:,2], marker="x")

        # final figure details
        ax.set_xlabel("$f_1(\mathbf{x})$", fontsize=15)
        ax.set_ylabel("$f_2(\mathbf{x})$", fontsize=15)
        ax.set_zlabel("$f_3(\mathbf{x})$", fontsize=15)
        ax.view_init(elev=11, azim=-25)
        ax.axes.set_xlim3d(left=0, right=1) 
        ax.axes.set_ylim3d(bottom=0, top=1) 
        ax.axes.set_zlim3d(bottom=0, top=1) 
        #ax.autoscale(tight=True)
        
        plt.legend()
        plt.tight_layout()
        plt.show()
    


    def ObjFun2D(ref_points, pop_fit, best_idx=None):

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)

        # Coordinate origin
        ax.scatter(0, 0, c="k", marker="+", s=100) 

        # Plot best solution
        if not best_idx==None:
            ax.scatter(pop_fit[best_idx,0], pop_fit[best_idx,1], marker='o', linewidths=1, facecolors='none', edgecolors='black', s=60) 
        
        # Plot ref_points
        ax.scatter(ref_points[:,0], ref_points[:,1], marker="o")

        # Plot last population fitness
        ax.scatter(pop_fit[:,0], pop_fit[:,1], marker="x")

        # final figure details
        ax.set_xlabel("$f_1(\mathbf{x})$", fontsize=15)
        ax.set_ylabel("$f_2(\mathbf{x})$", fontsize=15)
        ax.set_xlim(left=0, right=1) 
        ax.set_ylim(bottom=0, top=1) 
       
        plt.show()
        
        '''
        N = 500
        x = np.random.rand(N)
        y = np.random.rand(N)
        colors = (0,0,0)
        area = np.pi*3

        # Plot
        plt.scatter(x, y, s=area, c=colors, alpha=0.5)
        plt.title('Scatter plot pythonspot.com')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

        plt.legend()
        plt.tight_layout()

        plt.show()
        '''



    def MacroStressStrain(Simul_data_file, custom_dt_file, nsteps):

        # How to plot the macroscopic stress strain data (Robert Carson)

        font = {'size'   : 14}
        rc('font', **font)
        rc('mathtext', default='regular')

        # We can have differnt colors for our curves
        clrs = ['red', 'blue', 'green', 'black']
        mrks = ['*', ':', '--', 'solid']

        fig, ax = plt.subplots(1)

        # uncomment the below when the fileLoc is valid
        data = np.loadtxt(Simul_data_file, comments='%')
       
        # only here to have something that'll plot
        epsdot = 1e-3

        sig = data[:,1]
        # uncomment the below when the fileLoc is valid
        time = np.loadtxt(custom_dt_file)

        # only here to have something that'll plot
        time = np.ones(nsteps)
        eps = np.zeros(nsteps)

        for i in range(0, nsteps):
            dtime = time[i]
            if sig[i] - sig[i - 1] >= 0:
                eps[i] = eps[i - 1] + epsdot * dtime
            else:
                eps[i] = eps[i - 1] - epsdot * dtime

        ax.plot(eps, sig, 'r')
        ax.grid()

        # change this to fit your data                 
        # ax.axis([0, 0.01, 0, 0.3])

        ax.set_ylabel('Macroscopic engineering stress [GPa]')
        ax.set_xlabel('Macroscopic engineering strain [-]')

        fig.show()
        plt.show()
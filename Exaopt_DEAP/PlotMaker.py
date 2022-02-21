import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from deap import tools

class ExaPlots:
    def __init__(self, last_pop_fit, ref_points):
        self.last_pop_fit = last_pop_fit
        self.ref_points = ref_points

    def ObjFun3D(self):
        #NOBJ = 3
        #P = [2, 1]
        #SCALES = [1, 0.5]

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")

        # the coordinate origin
        ax.scatter(0, 0, 0, c="k", marker="+", s=100)
        ax = fig.add_subplot(111,projection='3d') 

        # reference points
        # Parameters
        #NOBJ = 3
        #P = [2, 1]
        #SCALES = [1, 0.5]

        # Create, combine and removed duplicates
        #ref_points = [tools.uniform_reference_points(NOBJ, p, s) for p, s in zip(P, SCALES)]
        #ref_points = numpy.concatenate(ref_points, axis=0)
        #_, uniques = numpy.unique(ref_points, axis=0, return_index=True)
        #ref_points = ref_points[uniques]
        ##
        #for subset, p, s in zip(ref_points, P, SCALES):
        # ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2], marker="o", s=48, label="p = {}, scale = {}".format(p, s))
        ax.scatter(self.ref_points[:,0],self.ref_points[:,1],self.ref_points[:,2], marker="o")
        ax.scatter(self.last_pop_fit[:,0],self.last_pop_fit[:,1],self.last_pop_fit[:,2], marker="x")

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
    

    def ObjFun2D(self):

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)

        # The coordinate origin
        ax.scatter(0, 0, c="k", marker="+", s=100) 

        # Scatter plots
        ax.scatter(self.ref_points[:,0],self.ref_points[:,1], marker="o")
        ax.scatter(self.last_pop_fit[:,0],self.last_pop_fit[:,1], marker="x")

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
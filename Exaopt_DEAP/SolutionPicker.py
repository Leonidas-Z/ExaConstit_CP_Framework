from math import dist
import numpy as np
import autograd.numpy as anp


class BestSol:

    # Except for ASF, all the below ways they use as a utopian point the (0,0) and the minimization strategy

    def __init__(self, pop_fit, weights=None, normalize=False):

        # If weights are not given
        if weights == None:
            self.weights = np.array([1] * len(pop_fit[0]))
        else:
            self.weights = np.array(weights)
        
        # If normalize true
        if normalize == True:
            approx_ideal = pop_fit.min(axis=0)
            approx_nadir = pop_fit.max(axis=0)
            self.fit = (pop_fit - approx_ideal)/(approx_nadir - approx_ideal)
        else:
            self.fit = pop_fit


    def ASF(self):

        # ASF Decomposition Method
        # Multiply by weighs the fit_values and then pick the max for each row. Then pick the min
        # This way gives best solution based on the least max error in the solution's obj functions for the population
        asf = ((self.fit - 0) * self.weights).max(axis=1)
        best_idx = np.argmin(asf)

        return best_idx
        

    def EUDIST(self, p=2):

        # Calcualte Euclidean Weighted Distance (When p=2 then vector magnitude from the origin)
        dist = (np.sum(self.weights * self.fit**p, axis=1))**(1/p)
        best_idx = np.argmin(dist)

        return best_idx



    '''
    def DIST_WEIGHTS(self, weights, utopian_point):
        
        norm = anp.linalg.norm(weights, axis=1)
        self.pop_fit = self.pop_fit - utopian_point

        d1 = (self.pop_fit * weights).sum(axis=1) / norm
        d2 = anp.linalg.norm(self.pop_fit - (d1[:, None] * weights / norm[:, None]), axis=1)

        return d1, d2
    '''
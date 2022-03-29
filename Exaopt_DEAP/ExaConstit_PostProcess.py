from matplotlib.pyplot import grid
import numpy
import pickle
import sys


''' 
This script is using pymoo module visualization folder to show useful plots of the data 
'''


# Read file
output="checkpoint_files/output_gen_1.pkl"

# Retrieve the state of the specified checkpoint
with open(output,"rb+") as ckp_file:
    ckp = pickle.load(ckp_file)

pop_fit = ckp["pop_fit"]
pop_param = ckp["pop_param"]
pop_stress = ckp["pop_stress"]
iter_tot = ckp["iter_tot"]
last_gen = ckp["generation"]

# ================================ Post Processing ===================================
# Choose which generation you want to show in plots
gen = -1 # here we chose the last gen (best)
pop_fit = pop_fit[gen]  
pop_fit = numpy.array(pop_fit) 
NPOP = pop_fit.shape[0]
NONJ = pop_fit.shape[1]

#NOBJ = numpy.array(pop_fit).shape[2]

# Find best solution
from ExaConstit_SolPicker import BestSol
best_idx=BestSol(pop_fit, weights=[0.5, 0.5], normalize=False).EUDIST()

print(best_idx)

# Visualize the results (here we used the visualization module of pymoo extensively)
from visualization.ExaPlotLibrary import ExaPlots
# Note that: pop_stress[gen][ind][expSim][file]
# first dimension is the selected generation, 
# second is the selected individual, 
# third is if we want to use experiment [0] or simulation [1] data, 
# forth is the selected experiment file used for the simulation 
strain_rate=1e-3
for k in range(numpy.array(pop_stress).shape[3]):
    S_exp = pop_stress[gen][best_idx][0][k]
    S_sim = pop_stress[gen][best_idx][1][k]
    plot = ExaPlots.StressStrain(S_exp, S_sim, epsdot = strain_rate)

from visualization.scatter import Scatter
plot = Scatter(tight_layout=False)
plot.add(pop_fit, s=20)
plot.add(pop_fit[best_idx], s=30, color="red")
plot.show()

from visualization.pcp import PCP
plot = PCP(tight_layout=False)
plot.set_axis_style(color="grey", alpha=0.5)
plot.add(pop_fit, color="grey", alpha=0.3)
plot.add(pop_fit[best_idx], linewidth=2, color="red")
plot.show()

from visualization.petal import Petal
plot = Petal(bounds=[0, 0.02], tight_layout=False)
plot.add(pop_fit[best_idx])
plot.show()
#Put out of comments if we want to see all the individual fitnesses and not only the best
plot = Petal(bounds=[0, 0.02], title=["Sol %s" % t for t in range(0,NPOP)], tight_layout=False)
for k in range(1,NPOP+1):
    if k%4==0:
        plot.add(pop_fit[k-4:k])
plot.show()


'''
from visualization.stress import Stress
strain_rate=1e-3
plot = Stress(tight_layout=False, epsdot=1e-3)
S_exp = pop_stress[gen][best_idx][0]
S_sim = pop_stress[gen][best_idx][1]
plot.add(S_exp, plot_type="line")
plot.add(S_sim, plot_type="line")

#plot.add(pop_stress[gen][best_idx][1][0], plot_type="line")
plot.show()
'''
from matplotlib.pyplot import grid
import numpy
import pickle
import sys
from sklearn.feature_extraction import grid_to_graph


''' 
This script is using pymoo module visualization folder to show useful plots of the data 
'''


NPOP = 52
'''
# Read file
output="checkpoint_files/output_gen_50.pkl"

# Retrieve the state of the specified checkpoint
with open(output,"rb+") as ckp_file:
    ckp = pickle.load(ckp_file)

pop_fit = ckp["pop_fit"]
pop_param = ckp["pop_param"]
pop_stress = ckp["pop_stress"]
iter_tot = ckp["iter_tot"]
last_gen = ckp["generation"]
#NPOP = ckp["NPOP"]

print(sys.getsizeof(pop_stress))

# ================================ Post Processing ===================================
# Choose which generation you want to show in plots
gen = -1 # here we chose the last gen (best)
pop_fit = pop_fit[gen]  
pop_fit = numpy.array(pop_fit) 


# Find best solution
from ExaConstit_SolPicker import BestSol
best_idx=BestSol(pop_fit, weights=[0.5, 0.5], normalize=False).EUDIST()


# Visualize the results (here we used the visualization module of pymoo extensively)
from visualization.PlotMaker import ExaPlots
strain_rate=1e-3
# Note that: pop_stress[gen][ind][expSim][file]
# first dimension is the selected generation, 
# second is the selected individual, 
# third is if we want to use experiment [0] or simulation [1] data, 
# forth is the selected experiment file used for the simulation 
file=0
plot2 = ExaPlots.MacroStressStrain(Exper_data = pop_stress[gen][best_idx][0][file], Simul_data = pop_stress[gen][best_idx][1][file], epsdot = strain_rate)
file=1
plot3 = ExaPlots.MacroStressStrain(Exper_data = pop_stress[gen][best_idx][0][file], Simul_data = pop_stress[gen][best_idx][1][file], epsdot = strain_rate)
'''
popl=numpy.array([[1],[2],[3],[5],[3],[1]])
from visualization.scatter import Scatter
plot = Scatter(tight_layout=False, grid=True)
plot.add(popl, s=20)
#plot.add(pop_fit[best_idx], s=30, color="red")
plot.show()
'''
from visualization.pcp import PCP
plot = PCP(tight_layout=True)
plot.set_axis_style(color="grey", alpha=0.5)
plot.add(pop_fit, color="grey", alpha=0.3)
plot.add(pop_fit[best_idx], linewidth=2, color="red")
plot.show()

from visualization.petal import Petal
plot = Petal(bounds=[0, 0.02], tight_layout=False)
plot.add(pop_fit[best_idx])
plot.show()
#Put out of comments if we want to see all the individual fitnesses and not only the best
plot = Petal(bounds=[0, 0.02], title=["Sol %s" % t for t in range(1,NPOP+1)], tight_layout=False)
for k in range(1,NPOP+1):
    if k%4==0:
        plot.add(pop_fit[k-4:k])
plot.show()

'''
'''
print(pop_stress[gen][0][1][0])
print(pop_stress[gen][0][1][1])
print(pop_param[gen][0])
print(pop_param[gen][1])
'''
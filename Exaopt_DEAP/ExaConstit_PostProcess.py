import numpy
import pickle
from SolutionPicker import BestSol

NOBJ=2

# Read file
output="checkpoint_files/output_gen_11.pkl"


# Retrieve the state of the specified checkpoint
with open(output,"rb+") as ckp_file:
    ckp = pickle.load(ckp_file)

pop_fit = ckp["pop_fit"]
pop_param = ckp["pop_param"]
pop_stress = ckp["pop_stress"]
iter_tot = ckp["iter_tot"]
last_gen = ckp["generation"]
#NPOP = ckp["NPOP"]
NPOP = 8


# ================================ Post Processing ===================================
# Choose which generation you want to show in plots
gen = -1 # here we chose the last gen (best)
pop_fit = pop_fit[gen]  
pop_fit = numpy.array(pop_fit) 


# Find best solution
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

from visualization.scatter import Scatter
plot = Scatter(tight_layout=False)
plot.add(pop_fit, s=20)
plot.add(pop_fit[best_idx], s=30, color="red")

from visualization.pcp import PCP
plot = PCP(tight_layout=True)
plot.set_axis_style(color="grey", alpha=0.5)
plot.add(pop_fit, color="grey", alpha=0.3)
plot.add(pop_fit[best_idx], linewidth=2, color="red")
plot.show()

from visualization.petal import Petal
plot = Petal(bounds=[0, 0.02], tight_layout=True)
plot.add(pop_fit[best_idx])
plot.show()
#Put out of comments if we want to see all the individual fitnesses and not only the best
plot = Petal(bounds=[0, 0.02], title=["Sol %s" % t for t in range(1,NPOP+1)], tight_layout=True)
k = int(NPOP/2)
plot.add(pop_fit[:k])
plot.add(pop_fit[k:])
plot.show()
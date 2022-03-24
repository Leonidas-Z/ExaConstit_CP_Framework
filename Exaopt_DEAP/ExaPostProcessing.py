import numpy
import pickle
from SolutionPicker import BestSol
from deap import creator, base, tools, algorithms

NOBJ=2
# Create minimization problem (multiply -1 weights)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)
# Create the Individual class that it has also the fitness (obj function results) as a list
creator.create("Individual", list, fitness=creator.FitnessMin, stress=None)


# Read file
checkpoint="checkpoint_files/checkpoint_gen_50.pkl"


# Retrieve the state of the specified checkpoint
with open(checkpoint,"rb+") as ckp_file:
    ckp = pickle.load(ckp_file)

pop_fit = ckp["pop_fit"]
pop_param = ckp["pop_param"]
pop_stress = ckp["pop_stress"]
iter_tot = ckp["iter_tot"]
last_gen = ckp["generation"]


# ================================ Post Processing ===================================
# Choose which generation you want to show in plots
pop_fit = pop_fit[-1]  # here we chose the last gen (best)
pop_fit = numpy.array(pop_fit) 


# Find best solution
best_idx=BestSol(pop_fit, weights=[0.5, 0.5], normalize=False).EUDIST()


# Visualize the results
from visualization.PlotMaker import ExaPlots

# Note that: pop_stress[gen][ind][expSim][file]
# first dimension is the selected generation, 
# second is the selected individual, 
# third is if we want to use experiment [0] or simulation [1] data, 
# forth is the selected experiment file used for the simulation 
gen = last_gen 
ind = best_idx
strain_rate=1e-3
file=0
plot2 = ExaPlots.MacroStressStrain(Exper_data = pop_stress[gen][ind][0][file], Simul_data = pop_stress[gen][ind][1][file], epsdot = strain_rate)
file=1
plot3 = ExaPlots.MacroStressStrain(Exper_data = pop_stress[gen][ind][0][file], Simul_data = pop_stress[gen][ind][1][file], epsdot = strain_rate)

from visualization.scatter import Scatter
plot = Scatter()
plot.add(pop_fit)
plot.add(pop_fit[best_idx], s=30, color="red", )
#plot.add(ref_points)
plot.show()

from visualization.pcp import PCP
plot = PCP()
plot.set_axis_style(color="grey", alpha=0.5)
plot.add(pop_fit, color="grey", alpha=0.3)
plot.add(pop_fit[best_idx], linewidth=2, color="red")
plot.show()

from visualization.petal import Petal
Petal(bounds=[0, 0.05]).add(pop_fit[best_idx]).show()
#Put out of comments if we want to see all the individual fitnesses and not only the best
#print(len(pop_fit))
#plot = Petal(bounds=[0, 0.05], title=["Sol %s" % t for t in range(1,MU+1)])
#plot.add(pop_fit[:MU])
#plot.show()


Simul_data = pop_stress[gen][ind][0][0]
print(Simul_data)
Simul_data = pop_stress[gen][ind][0][1]
print(Simul_data)
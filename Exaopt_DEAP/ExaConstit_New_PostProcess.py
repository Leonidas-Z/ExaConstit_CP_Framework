import pickle 
import random
from DEAP_mod import creator, base
import numpy


NOBJ = 2
GEN = 2
checkpoint="adsfasfd"


def PostProcess(pop_lib=None, checkpoint=None, NOBJ=NOBJ):

    # Create minimization problem (multiply -1 weights)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)
    # Create the Individual class that it has also the fitness (obj function results) as a list
    creator.create("Individual", list, fitness=creator.FitnessMin, rank=None, stress=None)

    if pop_lib==None and checkpoint==None:
        raise "No inputs provided"
        
    elif not checkpoint==None:

        with open(checkpoint,"rb+") as ckp_file:
                ckp = pickle.load(ckp_file)
        try:
            # Retrieve the state of the last checkpoint
            pop_lib = ckp["population_library"]
        except:
            raise "Could not read checkpoint file"

    # Retrieve some more info
    NGEN = pop_lib.shape[0]
    NPOP = pop_lib.shape[1]

    # Extract useful data from the pop class
    pop_fit_gen=[]
    pop_par_gen=[]
    pop_stress_gen=[]
    best_front_gen=[]
    best_front_par_gen=[]
    best_front_fit_gen=[]
    best_front_stress_gen=[]     
    pop_fit = []
    pop_param = []
    pop_stress = []
    # For gen=0 we dont do selection thus there is no best_front
    best_front_par = [[None]]
    best_front_fit = [[None]] 

    for gen in range(NGEN):
        for ind in pop_lib[gen]:
            pop_fit_gen.append(ind.fitness.values)
            pop_par_gen.append(tuple(ind))
            pop_stress_gen.append(ind.stress)

            if not gen == 0:
                if ind.rank == 0:
                    best_front_gen.append(ind)
                    best_front_fit_gen.append(ind.fitness.values)
                    best_front_par_gen.append(tuple(ind))
                    best_front_stress_gen.append(ind.stress)

        best_front_fit.append(best_front_fit_gen)
        best_front_par.append(best_front_par_gen)
        pop_fit.append(pop_fit_gen)
        pop_param.append(pop_par_gen)
        pop_stress.append(pop_stress_gen)

        # Make data numpy type (best_front has different size per generation, thus it is not so simple)
    pop_fit = numpy.array(pop_fit)



    # Find best solution
    from ExaConstit_SolPicker import BestSol
    best_idx = BestSol(pop_fit[GEN], weights=[1, 1]).EUDIST()


    # Visualize the results (here we used the visualization module of pymoo extensively)
    from Visualization.ExaPlots import StressStrain
    # Note that: pop_stress[gen][ind][expSim][file]
    # first dimension is the selected generation, 
    # second is the selected individual, 
    # third is if we want to use experiment [0] or simulation [1] data, 
    # forth is the selected experiment file used for the simulation 
    strain_rate=1e-3
    for k in range(numpy.array(pop_stress).shape[3]):
        S_exp = pop_stress[GEN][best_idx][0][k]
        S_sim = pop_stress[GEN][best_idx][1][k]
        plot = StressStrain(S_exp, S_sim, epsdot = strain_rate)


    from Visualization.scatter import Scatter
    plot = Scatter()
    plot.add(pop_fit[GEN], s=20)
    plot.add(numpy.array(best_front_fit[GEN]), s=20, color="orange")
    plot.add(pop_fit[GEN][best_idx], s=30, color="red")
    plot.show()


    from Visualization.pcp import PCP
    plot = PCP(tight_layout=False)
    plot.set_axis_style(color="grey", alpha=0.5)
    plot.add(pop_fit[GEN], color="grey", alpha=0.3)
    plot.add(pop_fit[GEN][best_idx], linewidth=2, color="red")
    plot.show()


    from Visualization.petal import Petal
    plot = Petal(bounds=[0, 0.05], tight_layout=False)
    plot.add(pop_fit[GEN][best_idx])
    plot.show()
    #Put out of comments if we want to see all the individual fitnesses and not only the best
    plot = Petal(bounds=[0, 0.05], title=["Sol %s" % t for t in range(0,NPOP)], tight_layout=False)
    for k in range(1,NPOP+1):
        if k%4==0:
            plot.add(pop_fit[GEN][k-4:k])
    plot.show()

    # VISUALIZATION
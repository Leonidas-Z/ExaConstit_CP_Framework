from matplotlib.pyplot import grid
import numpy
import pickle
import sys


        
''' 
This script is using pymoo module visualization folder to show useful plots of the data as a PostProcessing procedure and
it has no DEAP dependencies. 
This script can be used for any output file independantly of the optimization procedure. Therefore, it can be called 
during the optimization framework to watch its progress for a specified generation that has already been calculated.
How to run: You can call this function from any script or you can specify the inputs and run this script
'''


#========================= Inputs ==============================
gen = -1
output = "checkpoint_files/output_gen_6.pkl"




#====================== Post Processing ========================
def ExaPostProcessing(output="checkpoint_files/output_gen_1.pkl", gen=-1):

    # Retrieve the state of the specified output file
    with open(output,"rb+") as ckp_file:
        ckp = pickle.load(ckp_file)

    pop_fit = ckp["pop_fit"]
    pop_param = ckp["pop_param"]
    pop_stress = ckp["pop_stress"]
    best_front_fit = ckp["best_front_fit"]
    best_front_param = ckp["best_front_param"]
    iter_tot = ckp["iter_tot"]
    last_gen = ckp["generation"]

    # Make data numpy type (best_front has different size per generation, thus it is not so simple)
    pop_fit = numpy.array(pop_fit)

    # Retrieve some more info
    NGEN = pop_fit.shape[0]
    NPOP = pop_fit.shape[1]
    NOBJ = pop_fit.shape[2]


    # Find best solution
    from ExaConstit_SolPicker import BestSol
    best_idx = BestSol(pop_fit[gen], weights=[1, 1]).EUDIST()


    # Visualize the results (here we used the visualization module of pymoo extensively)
    from Visualization.ExaPlots import StressStrain
    # Note that: pop_stress[gen][ind][expSim][file]
    # first dimension is the selected generation, 
    # second is the selected individual, 
    # third is if we want to use experiment [0] or simulation [1] data, 
    # forth is the selected experiment file used for the simulation 
    strain_rate=1e-3
    for k in range(numpy.array(pop_stress).shape[3]):
        S_exp = pop_stress[gen][best_idx][0][k]
        S_sim = pop_stress[gen][best_idx][1][k]
        plot = StressStrain(S_exp, S_sim, epsdot = strain_rate)


    from Visualization.scatter import Scatter
    plot = Scatter()
    plot.add(pop_fit[gen], s=20)
    plot.add(numpy.array(best_front_fit[gen]), s=20, color="orange")
    plot.add(pop_fit[gen][best_idx], s=30, color="red")
    plot.show()


    from Visualization.pcp import PCP
    plot = PCP(tight_layout=False)
    plot.set_axis_style(color="grey", alpha=0.5)
    plot.add(pop_fit[gen], color="grey", alpha=0.3)
    plot.add(pop_fit[gen][best_idx], linewidth=2, color="red")
    plot.show()


    from Visualization.petal import Petal
    plot = Petal(bounds=[0, 0.05], tight_layout=False)
    plot.add(pop_fit[gen][best_idx])
    plot.show()
    #Put out of comments if we want to see all the individual fitnesses and not only the best
    plot = Petal(bounds=[0, 0.05], title=["Sol %s" % t for t in range(0,NPOP)], tight_layout=False)
    for k in range(1,NPOP+1):
        if k%4==0:
            plot.add(pop_fit[gen][k-4:k])
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

ExaPostProcessing(output=output, gen=gen)
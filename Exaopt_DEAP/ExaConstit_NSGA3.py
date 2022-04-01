from deap import creator, base, tools, algorithms
import numpy
import random
from math import factorial
import pickle
import sys

from pyrsistent import s

from ExaConstit_Problems import ExaProb
from ExaConstit_SolPicker import BestSol


''' 
ExaConstit Optimization Routine (LZ)

This script is using the NSGAIII algorithm from the DEAP module and intends
to optimize Crystal Plasticity Parameters calling the ExaConstit simulation
program via ExaProb class.

The method that the objective functions are calculated can be found in 
the evaluation function in the ExaProb class.

For details about how DEAP module works please look at their tutorial:
https://deap.readthedocs.io/en/master/index.html 
Also please look at the associated paper for the NSGAIII
'''


#============================== Input Parameters ================================
# Problem Parameters
# Number of obj functions
NOBJ = 2

# Specify independent per experiment data file parameters (e.g. athermal parameters)
# Specify their upper and their lower bounds
IND_LOW = [150, 100,  50, 1500, 1e-5, 1e-3, 1e-4, 1e-5, 1e-6]
IND_UP  = [200, 150, 100, 2500, 1e-3, 1e-1, 1e-2, 1e-3, 1e-4]
# Specify parameters are different (dependent) per experiment data file (e.g. thermal parameters). 
# If no dependent then set DEP_LOW = None, DEP_UP = None
# Specify their upper and their lower bounds
DEP_LOW = None      #[1e-4, 1e-5, 1e-6]
DEP_UP =  None      #[1e-2, 1e-3, 1e-4]
# Specify parameters that will not be optimized and are different (dependent) per experiment data file (e.g. the temperature).
# If no such parameters, set DEP_UNOPT = None. 
# How to use: DEP_UNOPT = [[file1], [file2], ...], where [fileN] = [param1, param2, ...] 
DEP_UNOPT = [[270, 1e-3],[300, 2e-3]]


# Final Bounds
BOUND_LOW = IND_LOW
BOUND_UP = IND_UP
if (DEP_LOW != None) and (DEP_UP != None):
    for i in range(NOBJ):
        BOUND_LOW.extend(DEP_LOW)
        BOUND_UP.extend(DEP_UP)
    n_dep = len(DEP_LOW)
else:
    n_dep = None # no dependent parameters

# Number of total parameters or dimensions or genes: NDIM = len(IND) + len(DEP)*NOBJ
NDIM = len(BOUND_LOW)

# Number of generation (e.g. If NGEN=2 it will perform the population initiation gen=0, and then gen=1 and gen=2. Thus, NGEN+1 generations)
NGEN = 20

# Make the reference points using the uniform_reference_points method (function is in the emo.py within the selNSGA3)
p = [50, 0]
scaling = [1, 0.5]

ref1 = tools.uniform_reference_points(NOBJ, p[0], scaling[0])
if p[1]!=0 and scaling[1]!=0:
    ref2 = tools.uniform_reference_points(NOBJ, p[1], scaling[1])
    ref_points=numpy.concatenate((ref1, ref2), axis=0)
else: 
    ref_points=ref1
    
# Number of Reference Points (NSGAIII paper)
P = sum(p)
H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))

# Population number (NSGAIII paper)
NPOP = int(H + (4 - H % 4))

# GA operator related parameters
CXPB = 1.0
MUTPB = 1.0

# Specify ExaProb class arguments to run ExaConstit simulations and evaluate the objective functions
problem = ExaProb(n_obj=NOBJ,
                  mult_GA=True,
                  n_dep=n_dep,
                  dep_unopt = DEP_UNOPT,
                  n_steps=[20,20],
                  ncpus = 20,
                  #loc_mechanics_bin ="",
                  Exper_input_files = ['Experiment_stress_270.txt', 'Experiment_stress_300.txt'],
                  Sim_output_files = ['test_mtsdd_bcc_stress.txt','test_mtsdd_bcc_stress.txt'],
                  Toml_files = ['./mtsdd_bcc.toml', './mtsdd_bcc.toml'],
                  )

# Specify seed (if checkpoint!=None it doesn't matter)
seed=None

# Specify checkpoint frequency (generations per checkpoint)
checkpoint_freq = 1

# Specify checkpoint file or set None if you want to start from the beginning
checkpoint= None #"checkpoint_files/checkpoint_gen_10.pkl"

# Specify how many simulation failures in total to have so to terminate the optimization framework
fail_limit = 5



print("\nNumber of objective functions = {}".format(NOBJ))
print("Number of parameters = {}".format(NDIM))
print("Number of generations = {}".format(NGEN))
print("Number of reference points = {}".format(H))
print("Population size = {}".format(NPOP))
print("Expected Total Iterations = {}".format(NPOP*NGEN))
print("Expected Total Simulation Runs = {}\n".format(NPOP*NOBJ*NGEN))



#====================== Initialize Optimization Strategy ======================
# Create minimization problem (multiply -1 weights)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)
# Create the Individual class that it has also the fitness (obj function results) as a list
creator.create("Individual", list, fitness=creator.FitnessMin, stress=None)



#=========================== Initialize Population ============================
# Generate a random individual with respect to his gene boundaries. 
# Low and Up can be columns with same size as the number of genes of the individual
def uniform(low, up, size=None):  
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]


#### Population generator
toolbox = base.Toolbox()
# Register the above individual generator method in the toolbox class. That is attr_float with arguments low=BOUND_LOW, up=BOUND_UP, size=NDIM
toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
# Function that produces a complete individual with NDIM number of genes that have Low and Up bounds
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
# Function that instatly produces MU individuals (population). We assign the attribute number_of_population at the main function in this problem
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


#=========================== Initialize GA Operators ============================
#### Evolution Methods
# Function that returns the objective functions values as a dictionary (if n_obj=3 it will evaluate the obj function 3 times and will return 3 values (str) - It runs the problem.evaluate for n_obj times)
toolbox.register("evaluate", problem.evaluate)   # Evaluate obj functions
# Crossover function using the cxSimulatedBinaryBounded method
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=30.0)
# Mutation function that mutates an individual using the mutPolynomialBounded method. A high eta will producea mutant resembling its parent, while a small eta will produce a ind_fit much more different.
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
# Selection function that selects individuals from population + offspring using selNSGA3 method (non-domination levels, etc (look at paper for NSGAIII))
toolbox.register("select", tools.selNSGA3, ref_points=ref_points)


#================================ Evolution Algorithm ===========================
# Here we construct our main algorithm NSGAIII
def main(seed=None, checkpoint=None, checkpoint_freq=1):

    # Initialize statistics object
    stats1 = tools.Statistics(lambda ind: ind.fitness.values)
    stats1.register("avg", numpy.mean, axis=0)
    stats1.register("std", numpy.std, axis=0)
    stats1.register("min", numpy.min, axis=0)
    stats1.register("max", numpy.max, axis=0)

    # If checkpoint has a file name, read and retrieve the state of last checkpoint from this file
    # If not then start from the beginning by generating the initial population
    if checkpoint:
        
        with open(checkpoint,"rb+") as ckp_file:
            ckp = pickle.load(ckp_file)
        
        try:
            # Retrieve random state
            random.setstate(ckp["rndstate"]) 

            # Retrieve the state of the last checkpoint
            pop = ckp["population"]
            pop_fit = ckp["pop_fit"]
            pop_param = ckp["pop_param"]
            pop_stress = ckp["pop_stress"]
            fail_count = ckp["fail_count"]
            iter_tot = ckp["iter_tot"]
            start_gen = ckp["generation"] + 1
            if start_gen>NGEN: gen = start_gen
            logbook1 = ckp["logbook1"]
            logbook2 = ckp["logbook2"]
        except:
            print("\nERROR: Wrong Checkpoint file")
            sys.exit()

        # Open log files and erase their contents
        logfile1 = open("logbook1_stats.log","w+")
        logfile1.write("loaded checkpoint: {}\n".format(checkpoint))
        logfile2 = open("logbook2_solutions.log","w+")
        logfile2.write("loaded checkpoint: {}\n".format(checkpoint))

    else:
        # Specify seed (need both numpy and random OR change niching in DEAP script)
        random.seed(seed)  
        numpy.random.seed(seed) 
        
        # Initialize loggers
        logbook1 = tools.Logbook()
        logfile1 = open("logbook1_stats.log","w+")
        logbook2 = tools.Logbook()
        logfile2 = open("logbook2_solutions.log","w+")

        # Initialize counters and lists
        iter_pgen = 0       # iterations per generation
        iter_tot = 0        # total iterations
        fail_count = 1
        start_gen = 1       # starting generation
        pop_fit = []
        pop_param = []
        pop_stress = []

        # Produce initial population
        # We use the registered "population" method MU times and produce the population
        pop = toolbox.population(n=NPOP)                                
        
        # Returns the individuals with an invalid fitness
        # invalid_ind is a list with NDIM genes in col and invalid_ind IDs in rows)
        # Maps the fitness with the invalid_ind. Initiates the obj function calculation for each invalid_ind
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        len_invalid_ind = len(invalid_ind)

        fitness_eval = toolbox.map(toolbox.evaluate, invalid_ind)

        # Evaluates the fitness for each invalid_ind and assigns them the new values
        for k ,ind, fit in zip(range(len(invalid_ind)), invalid_ind, fitness_eval): 
            iter_pgen+=1
            iter_tot+=1
#_______________________________________________________________________________________________
            # If simulation failed due to parameters, pick randomly 2 indivuduals, mate and mutate and try to run simulation with the new individual
            # Stopping criteria: If there is more than fail_limit number of simulation failures, terminate framework
            if not problem.is_simulation_done() == 0:

                while fail_count <= fail_limit:
                    text="Attempt to find another Parameter set to converge, fail_count = {}\n\n".format(fail_count)
                    problem.write_ExaProb_log(text, "warning", changeline=False)
                    fail_count+=1
                    # Need 2 different individuals to apply mate and mutate and derive a new individual
                    ind_idx = random.sample(range(0, len_invalid_ind), 2)
                    ind1 = invalid_ind[ind_idx[0]]
                    ind2 = invalid_ind[ind_idx[1]]
                    # Replace old individual with the new one with the hope that now the simulation will run normally
                    ind[:] = toolbox.mate(ind1, ind2)[0]             
                    ind[:] = toolbox.mutate(ind)[0]
                    # Run simulation to find the obj functions
                    fit = toolbox.evaluate(ind)
                    # If simulation successful break the loop 
                    if problem.is_simulation_done() == 0: 
                        break
                else:
                    text = "The evaluation failed for a total of {} attempts! Framework will terminate!".format(fail_count-1)
                    problem.write_ExaProb_log(text, "error", changeline=True)
                    sys.exit()

            ind.fitness.values = fit
            ind.stress = problem.return_stress()
#_______________________________________________________________________________________________

        # Write log statistics about the new population
        logbook1.header = "gen", "iter", "simRuns", "std", "min", "avg", "max"
        record = stats1.compile(pop)
        logbook1.record(gen=0, iter=iter_pgen, simRuns=iter_pgen*NOBJ, **record)
        logfile1.write("{}\n".format(logbook1.stream))
        
        # Write log file and store important data
        pop_fit_gen = []
        pop_par_gen = []
        pop_stress_gen = []
        logbook2.header = "gen", "fitness", "solutions"
        for ind in pop:
            logbook2.record(gen=0, fitness=list(ind.fitness.values), solutions=list(ind))
            logfile2.write("{}\n".format(logbook2.stream))
            # Save data
            pop_fit_gen.append(ind.fitness.values)
            pop_par_gen.append(tuple(ind))
            pop_stress_gen.append(ind.stress)

        # Keep fitnesses, solutions and stress for every gen in a list
        pop_fit.append(pop_fit_gen)
        pop_param.append(pop_par_gen)
        pop_stress.append(pop_stress_gen)


    # Begin the generational process
    for gen in range(start_gen, NGEN+1):
          
        logfile1 = open("logbook1_stats.log","a+")
        logfile2 = open("logbook2_solutions.log","a+")

        # Produce offsprings
        # varAnd does the previously registered crossover and mutation methods. 
        # Produces the offsprings and deletes their previous fitness values
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)    
                   
        # Evaluate the individuals that their fitness has not been evaluated
        # Returns the invalid_ind (in each row, returns the genes of each invalid_ind). 
        # Invalid_ind are those which their fitness value has not been calculated 
        # Evaluates the obj functions for each invalid_ind (here we have 3 obj function thus it does 3 computations)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]     
        fitness_eval = toolbox.map(toolbox.evaluate, invalid_ind)                

        # Assign the new values in the individuals
        iter_pgen = 0 
        for ind, fit in zip(invalid_ind, fitness_eval):            
            iter_pgen+=1
            iter_tot+=1
#_______________________________________________________________________________________________
            # If simulation failed due to parameters, pick randomly 2 indivuduals, mate and mutate and try to run simulation with the new individual
            # Stopping criteria: If there is more than fail_limit number of simulation failures, terminate framework
            if not problem.is_simulation_done() == 0:

                while fail_count <= fail_limit:
                    text="Attempt to find another Parameter set to converge, fail_count = {}\n\n".format(fail_count)
                    problem.write_ExaProb_log(text, "warning", changeline=False)
                    fail_count+=1
                    # Need 2 different individuals to apply mate and mutate and derive a new individual
                    ind_idx = random.sample(range(0, len_invalid_ind), 2)
                    ind1 = invalid_ind[ind_idx[0]]
                    ind2 = invalid_ind[ind_idx[1]]
                    # Replace old individual with the new one with the hope that now the simulation will run normally
                    ind[:] = toolbox.mate(ind1, ind2)[0]             
                    ind[:] = toolbox.mutate(ind)[0]
                    # Run simulation to find the obj functions
                    fit = toolbox.evaluate(ind)
                    # If simulation successful break the loop   
                    if problem.is_simulation_done() == 0: 
                        break
                else:
                    text = "The evaluation failed for a total of {} attempts! Framework will terminate!".format(fail_count-1)
                    problem.write_ExaProb_log(text, "error", changeline=True)
                    sys.exit()

            ind.fitness.values = fit
            ind.stress = problem.return_stress()
#_______________________________________________________________________________________________

        # Select (selNSGAIII) MU individuals as the next generation population from pop+offspring
        # In selection, random does not follow the rules because in DEAP, NSGAIII niching is using numpy.random() and not random.random() !!!!! 
        # Please change to random.shuffle
        pop = toolbox.select(pop + offspring, NPOP)                            

        # Write log statistics about the new population
        record = stats1.compile(pop)
        logbook1.record(gen=gen, iter=iter_pgen, simRuns=iter_pgen*NOBJ, **record)
        logfile1.write("{}\n".format(logbook1.stream))


        # Write log file and store important data
        pop_fit_gen=[]
        pop_par_gen=[]
        pop_stress_gen=[]
        for ind in pop: 
            logbook2.record(gen=gen, fitness=list(ind.fitness.values), solutions=list(ind))
            logfile2.write("{}\n".format(logbook2.stream))
            # Save data
            pop_fit_gen.append(ind.fitness.values)
            pop_par_gen.append(tuple(ind))
            pop_stress_gen.append(ind.stress)

        # Keep fitnesses, solutions and stress for every gen in a list
        pop_fit.append(pop_fit_gen)
        pop_param.append(pop_par_gen)
        pop_stress.append(pop_stress_gen)

        # Generate a checkpoint and output files (the output file will be independent of DEAP module)
        if gen % checkpoint_freq == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            ckp = dict(population=pop, pop_fit = pop_fit, pop_param=pop_param, pop_stress=pop_stress, iter_tot=iter_tot,\
                generation=gen, fail_count=fail_count, logbook1=logbook1, logbook2=logbook2, rndstate=random.getstate())
            with open("checkpoint_files/checkpoint_gen_{}.pkl".format(gen), "wb+") as ckp_file:
                pickle.dump(ckp, ckp_file)
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            out = dict(pop_fit = pop_fit, pop_param=pop_param, pop_stress=pop_stress, iter_tot=iter_tot, generation=gen)
            with open("checkpoint_files/output_gen_{}.pkl".format(gen), "wb+") as out_file:
                pickle.dump(out, out_file)
    
    logfile1.close()
    logfile2.close()

    return iter_tot, pop_fit, pop_param, pop_stress


# Call the optimization routine
iter_tot, pop_fit, pop_param, pop_stress = main(seed, checkpoint, checkpoint_freq)






#================================ Post Processing ===================================
# Choose which generation you want to show in plots
gen = -1 # here we chose the last gen (best)
pop_fit = pop_fit[gen]  
pop_fit = numpy.array(pop_fit) 


# Find best solution
best_idx=BestSol(pop_fit, weights=[0.5, 0.5], normalize=False).EUDIST()


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
plot.add(ref_points, color="blue")
plot.show()
plot = Scatter()
plot.add(pop_fit, s=20)
plot.add(pop_fit[best_idx], s=30, color="red")
plot.show()
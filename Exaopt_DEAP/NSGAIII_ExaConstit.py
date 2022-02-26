from deap import creator, base, tools, algorithms
import numpy
import random
from math import factorial
import pickle
from ExaConstit_problems import ExaProb
from PlotMaker import ExaPlots
from SolutionPicker import BestSol

##### ExaConstit Optimization Routine (LZ) #####

# This script is using the NSGAIII algorithm from the DEAP module and intends
# to optimize Crystal Plasticity Parameters calling the ExaConstit simulation 
# program via ExaProb class.

# The method that the objective functions are calculated can be found in 
# the evaluation function in the ExaProb class.

# For details about how DEAP module works please look at their tutorial:
# https://deap.readthedocs.io/en/master/index.html 
# Also please look at the associated paper for the NSGAIII



#============================== Input Parameters ================================
# Problem Parameters
# Number of obj functions
NOBJ = 2    
# The dependent parameters are included (the first 3 are the only independent)
BOUND_LOW =[1500, 1.0e-4, 1e-2   , 1e-3,    1e-4   , 1e-5    , 1e-3,    1e-4   , 1e-5]
BOUND_UP  =[2500, 10e-4 , 10.0e-2, 10.0e-3, 10.0e-4, 10.0e-5 , 10.0e-3, 10.0e-4, 10.0e-5]
# Number of parameters or dimensions or genes
NDIM = len(BOUND_LOW) 


# Assign ExaProb class and assign the arguments for the ExaConstit simulations
# Which of them are dependent
problem = ExaProb(n_obj=NOBJ,
                  n_steps=[20,20],
                  #x_dep=[],
                  #x_indep=[],
                  ncpus = 3,
                  #loc_mechanics_bin ="",
                  Exper_data_files = ['Experiment_stress_270.txt', 'Experiment_stress_300.txt'],
                  Toml_files = ['./mtsdd_bcc.toml', './mtsdd_bcc.toml'],
                  Simul_data_files = ['test_mtsdd_bcc_stress.txt','test_mtsdd_bcc_stress.txt'])


# Parameters related with Reference Points 
P = 30
# Scaling (None or 1 is the same)
scaling = None 

# Number of Reference Points (NSGAIII paper)
H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))  
print("\nThe number of reference points will be {}\n".format(H))
# Make the reference points using the uniform_reference_points method (function is in the emo.py within the selNSGA3)
ref_points = tools.uniform_reference_points(NOBJ, P, scaling)

# Population number (NSGAIII paper)
MU = int(H + (4 - H % 4)); print("\nThe population number will be {}\n".format(MU))

# Number of generation (e.g. If NGEN=2 it will perform the population initiation gen=0, and then gen=1 and gen=2. Thus, NGEN+1 generations)
NGEN = 10                

# Operators related parameters
CXPB = 1.0
MUTPB = 1.0

# Specify seed (if use checkpoint it doesn't matter)
seed=10

# Specify checkpoint frequency (generations per file)
checkpoint_freq = 1 
# Specify checkpoint file or set None if you want to start from the beginning
checkpoint = None#"checkpoint_files/checkpoint_gen_6.pkl"



#====================== Initialize Optimization Strategy ==========================
# Create minimization problem (multiply -1 weights)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)
# Create the Individual class that it has also the fitness (obj function results) as a list
creator.create("Individual", list, fitness=creator.FitnessMin)



#=========================== Initialize Population ============================
# Generate a random individual with respect to his gene boundaries. 
# Low and Up can be columns with same size as the number of genes of the individual
def uniform(low, up, size=None):  
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
#print(uniform(BOUND_LOW, BOUND_UP))
#problem.evaluate(uniform(BOUND_LOW, BOUND_UP))


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
# Mutation function that mutates an individual using the mutPolynomialBounded method. A high eta will producea mutant resembling its parent, while a small eta will produce a ind_fitution much more different.
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
# Selection function that selects individuals from population + offspring using selNSGA3 method (non-domination levels, etc (look at paper for NSGAIII))
toolbox.register("select", tools.selNSGA3, ref_points=ref_points)



#================================ Evolution Algorithm ===========================
# Here we construct our main algorithm NSGAIII
def main(checkpoint_freq, checkpoint=None, seed=None):

    # Initialize statistics object
    stats2 = tools.Statistics()
    stats2.register("fit", list)
    stats1 = tools.Statistics(lambda ind: ind.fitness.values)
    stats1.register("avg", numpy.mean, axis=0)
    stats1.register("std", numpy.std, axis=0)
    stats1.register("min", numpy.min, axis=0)
    stats1.register("max", numpy.max, axis=0)


    # If checkpoint has a file name, read and retrieve the state of last checkpoint from this file
    # If not then start from the beginning by generating the initial population
    if checkpoint:
        with open(checkpoint,"rb") as ckp_file:
            ckp = pickle.load(ckp_file)

        # Retrieve random state
        random.setstate(ckp["rndstate"]) 

        # Retrieve the state of the last checkpoint
        pop = ckp["population"]
        iter_tot = ckp["iter_tot"]
        start_gen = ckp["generation"] + 1
        if start_gen>NGEN: gen = start_gen
        logbook1 = ckp["logbook1"]
        logbook2 = ckp["logbook2"]
        # Open log files and erase their contents
        logfile1 = open("logbook1_gen.log","w+")
        logfile2 = open("logbook2_iter.log","w+")
        logfile1.truncate(0) 
        logfile2.truncate(0)
        logfile1.write("Loaded checkpoint: {}\n".format(checkpoint))
        logfile2.write("Loaded checkpoint: {}\n".format(checkpoint))

    else:
        # Specify seed (need both numpy and random OR change niching in DEAP script)
        random.seed(seed)  
        #numpy.random.seed(seed) 

        # Initialize logs and open log files
        logbook1 = tools.Logbook()
        logbook2 = tools.Logbook()
        logbook1.header = "gen", "runs", "std", "min", "avg", "max"
        logbook2.header = "gen", "iter_pgen", "iter_tot", "fit"
        logfile1 = open("logbook1_gen.log","w+")
        logfile2 = open("logbook2_iter.log","w+")
        logfile1.truncate(0)
        logfile2.truncate(0)

        # Produce initial population
        # We use the registered "population" method MU times and produce the population
        pop = toolbox.population(n=MU)                                
        
        # Returns the individuals with an invalid fitness
        # invalid_ind is a list with NDIM genes in col and invalid_ind IDs in rows)
        # Maps the fitness with the invalid_ind. Initiates the obj function calculation for each invalid_ind
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]   
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)        
        
        # Initialize iterations
        iter_pgen = 0  # iterations per generation
        iter_tot = 0   # total iterations
        start_gen = 1  # starting generation

        # Evaluates the fitness for each invalid_ind and assigns them the new values
        for ind, fit in zip(invalid_ind, fitnesses): 
            iter_pgen+=1
            iter_tot+=1
            ind.fitness.values = fit
            # Compile stastics about the obj. function and store to files
            record = stats2.compile(ind.fitness.values)
            logbook2.record(gen=0, iter_pgen=iter_pgen, iter_tot=iter_tot, **record)
            logfile2.write("{}\n".format(logbook2.stream))
        
        # Compile statistics about the population
        record = stats1.compile(pop)
        logbook1.record(gen=0, evals=len(invalid_ind), **record)
        logfile1.write("{}\n".format(logbook1.stream))


    # Begin the generational process
    for gen in range(start_gen, NGEN+1):

        # Produce offsprings
        # varAnd does the previously registered crossover and mutation methods. 
        # Produces the offsprings and deletes their previous fitness values
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)    
                   
        # Evaluate the individuals that their fitness has not been evaluated
        # Returns the invalid_ind (in each row, returns the genes of each invalid_ind). 
        # Invalid_ind are those which their fitness value has not been calculated 
        # Evaluates the obj functions for each invalid_ind (here we have 3 obj function thus it does 3 computations)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]     
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)                

        # Assign the new values in the individuals
        iter_pgen = 0 
        for ind, fit in zip(invalid_ind, fitnesses):                      
            iter_pgen+=1
            iter_tot+=1
            ind.fitness.values = fit
            # Compile stastics about the obj. function and store to files
            record = stats2.compile(ind.fitness.values)
            logbook2.record(gen=gen, iter_pgen=iter_pgen, iter_tot=iter_tot, **record)
            logfile2.write("{}\n".format(logbook2.stream))

        # Select (selNSGAIII) MU individuals as the next generation population from pop+offspring
        # In selection, random does not follow the rules because in DEAP, NSGAIII niching is using numpy.random() and not random.random() !!!!! 
        # Please change to random.shuffle
        pop = toolbox.select(pop + offspring, MU)                            

        # Compile statistics about the new population
        record = stats1.compile(pop)
        logbook1.record(gen=gen, evals=len(invalid_ind), **record)
        logfile1.write("{}\n".format(logbook1.stream))

        # Generate a checkpoint file
        if gen % checkpoint_freq == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            ckp = dict(population=pop, iter_tot=iter_tot, generation=gen, logbook1=logbook1, logbook2=logbook2, rndstate=random.getstate())
            with open("checkpoint_files/checkpoint_gen_{}.pkl".format(gen), "wb") as ckp_file:
                pickle.dump(ckp, ckp_file)

    logfile1.close()
    logfile2.close()

    return pop, gen, iter_tot


# Call the optimization routine
last_pop, gen, iter_tot = main(seed, checkpoint, checkpoint_freq)



#================================ Post Processing ===================================
# Save the objective function values of the last (best) population
pop_fit = []
for ind, k in zip(last_pop, range(len(last_pop))):
    pop_fit.append(ind.fitness.values)

pop_fit = numpy.array(pop_fit) 


# Find best solution
best_idx=BestSol(pop_fit, weights=[0.5, 0.5], normalize=False).EUDIST()


# Make a Plot
if NOBJ == 2:
    plot1 = ExaPlots.ObjFun2D(ref_points, pop_fit, best_idx)
elif NOBJ == 3:
    plot1 = ExaPlots.ObjFun3D(ref_points, pop_fit, best_idx)
else:
    pass

#print(pop_fit)
#print(best_idx)
#print(pop_fit[best_idx])
#print(numpy.shape(last_pop))
#plot2 = plot.MacroStressStrain(Simul_data_file='test_mtsdd_bcc_stress.txt', custom_dt_file='custom_dt.txt', nsteps=20)

from cmath import inf
from pstats import Stats
from deap import creator, base, tools, algorithms

import numpy
import random
from math import factorial
import matplotlib.pyplot as plt

from ExaConstit_problems import ExaProb
from PlotMaker import ExaPlots

#============================== Input Parameters ================================
#### Problem Parameters
NOBJ = 2
NDIM = 9   # n_var

# Number of reference points H (like in the NSGAIII paper)
P = 2
H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))  
print("\nThe number of reference points will be {}\n".format(H))

# The dependent parameters are included (the first 3 are the only independent)
BOUND_LOW =[1500, 1.0e-4, 1e-2   , 1e-3,    1e-4   , 1e-5    , 1e-3,    1e-4   , 1e-5]
BOUND_UP  =[2500, 10e-4 , 10.0e-2, 10.0e-3, 10.0e-4, 10.0e-5 , 10.0e-3, 10.0e-4, 10.0e-5]


#### Assign ExaProb class and assign the arguments for the ExaConstit simulations
# Which of them are dependent
# x_dep = {File1: "dsf"} (TBD)
problem = ExaProb(n_var=NDIM,
                  n_obj=NOBJ,
                  x_dep=[],
                  x_indep=[],
                  ncpus = 4,
                  options_toml = 'mtsdd_bcc.toml',
                  Simul_data_file = 'test_mtsdd_bcc_stress.txt',
                  Exper_data_files = ['Experiment_stress_270.txt', 'Experiment_stress_300.txt'])


#### Algorithm Parameters
# Population number as it was calculated in NSGAIII paper
MU = int(H + (4 - H % 4))
print("\nThe population number will be {}\n".format(MU))
NGEN = 2
CXPB = 1.0
MUTPB = 1.0



#========================== Initiate Optimization Problem ============================
# Make the reference points using uniform_reference_points method (function is in the emo.py with the selNSGA3)
ref_points = tools.uniform_reference_points(NOBJ, P, scaling = 0.5)

# Create minimization problem (multiply -1 weights)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)
# Create the Individual class that it has also the fitness (obj function results) as a list
creator.create("Individual", list, fitness=creator.FitnessMin)

# Generate a random individual with respect to his gene boundaries. Low and Up can be columns with same size as the number of genes of the individual
def uniform(low, up, size=None):  
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

#print(uniform(BOUND_LOW, BOUND_UP))
#problem.evaluate(uniform(BOUND_LOW, BOUND_UP))



#=========================== Initialize Population ============================
toolbox = base.Toolbox()

#### Population generator
# Register the above individual generator method in the toolbox class. That is attr_float with arguments low=BOUND_LOW, up=BOUND_UP, size=NDIM
toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
# Function that produces a complete individual with NDIM number of genes that have Low and Up bounds
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
# Function that instatly produces MU individuals (population). We assign the attribute number_of_population at the main function in this problem
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

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
def main(seed=None):
    random.seed(seed)

    # Initialize statistics object
    stats2 = tools.Statistics()
    stats2.register("fit", list)
    stats1 = tools.Statistics(lambda ind: ind.fitness.values)
    stats1.register("avg", numpy.mean, axis=0)
    stats1.register("std", numpy.std, axis=0)
    stats1.register("min", numpy.min, axis=0)
    stats1.register("max", numpy.max, axis=0)

    # Initialize logs and logfiles
    logbook2 = tools.Logbook()
    logbook2.header = "gen", "iter_pgen", "iter_tot", "fit"
    logfile2 = open("Logbook1_Iterations.log","w")
    logfile2.truncate(0)
    logbook1 = tools.Logbook()
    logbook1.header = "gen", "evals", "std", "min", "avg", "max"
    logfile1 = open("Logbook2_Generations.log","w")
    logfile1.truncate(0)

    # Produce population
    pop = toolbox.population(n=MU)                                # We use the registered "population" method MU times and produce the population
    
    # Returns the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]   # Returns the invalid_ind (list with the genes in col and invalid_ind IDs in rows). Fitness is invalid when it contains no values
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)        # Maps the fitness with the invalid_ind. Initiates the obj functions for each invalid_ind
    #print(toolbox.evaluate(invalid_ind[0]))

    # Evaluates the fitness for each invalid_ind and assigns them the new values
    iter_pgen = 0  # Iternations per generation
    iter_tot = 0   # Total generations
    for ind, fit in zip(invalid_ind, fitnesses): 
        ind.fitness.values = fit
        # Keep track of iterations
        iter_pgen+=1
        iter_tot+=1
        # Record stastics of obj. fun and store to files
        record = stats2.compile(ind.fitness.values)
        logbook2.record(gen=0, iter_pgen=iter_pgen, iter_tot=iter_tot, **record)
        logfile2.write("{}\n".format(logbook2.stream))
    
    # Compile statistics about the population
    record = stats1.compile(pop)
    logbook1.record(gen=0, evals=len(invalid_ind), **record)
    logfile1.write("{}".format(logbook1.stream))

    # Begin the generational process
    for gen in range(1, NGEN):
        
        # Produce offsprings
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)               # varAnd does the previously registered crossover and mutation methods. Produces the offsprings and deletes their previous fitness values

        # Evaluate the individuals that their fitness has not been evaluated 
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]     # Returns the invalid_ind (in each row, returns the genes of each invalid_ind). Invalid_ind are those which their fitness value has not been calculated
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)                # Evaluates the obj functions for each invalid_ind (here we have 3 obj function thus it does 3 computations)
       
        # Assign the new values in the individuals
        iter_pgen = 0 
        for ind, fit in zip(invalid_ind, fitnesses):                      
            ind.fitness.values = fit
            # Keep track of iterations
            iter_pgen+=1
            iter_tot+=1
            # Record stastics of obj. fun and store to files
            record = stats2.compile(ind.fitness.values)
            logbook2.record(gen=gen, iter_pgen=iter_pgen, iter_tot=iter_tot, **record)
            logfile2.write("{}\n".format(logbook2.stream))

        # Select (NSGA-III) MU individuals as the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, MU)                             # select does the previously registered seletion method selNSGAIII to the pop+offspring. Generates new pop

        # Compile statistics about the new population
        record = stats1.compile(pop)
        logbook1.record(gen=gen, evals=len(invalid_ind), **record)
        logfile1.write("\n{}".format(logbook1.stream))

    logfile1.close()
    logfile2.close()

    return pop, gen #, logbook1, iter_tot #, fitnessval

# Call the optimization routine
last_pop, last_gen = main()



#================================ Post Processing ===================================
# Save the objective function values of the last (best) population
last_pop_fit = []
for ind, k in zip(last_pop, range(len(last_pop))):
    last_pop_fit.append(ind.fitness.values)

last_pop_fit = numpy.array(last_pop_fit) 
#print(last_pop_fit)


# Find Best ind_fitution from the last_pop_fit using vector magnitude
best_magn = inf
ind_ID = 0
for ind_fit, k in zip(last_pop_fit, range(len(last_pop_fit))):
    magn = numpy.sqrt(numpy.sum(ind_fit**2))
    ind_ID+=1
    if magn < best_magn:
        best_magn = magn
        best_ind_fit = ind_fit
        best_ID = ind_ID

print(last_pop_fit)  
print(best_ID)   
print(best_ind_fit)



# Make a Plot
plot = ExaPlots(last_pop_fit, ref_points)
if NOBJ == 2:
    plot1 = plot.ObjFun2D()
elif NOBJ == 3:
    plot1 = plot.ObjFun3D()
else:
    pass

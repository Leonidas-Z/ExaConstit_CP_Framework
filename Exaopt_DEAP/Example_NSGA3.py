from DEAP_mod import creator, base, tools, algorithms
from DEAP_mod.benchmarks.tools import hypervolume, convergence, diversity
from math import factorial
import random

import matplotlib.pyplot as plt
import numpy
import pymop.factory
from deap.benchmarks.tools import igd
from ExaConstit_SolPicker import BestSol
from Visualization import ExaPlots


UNSGA3 = True
# Problem definition
PROBLEM = "dtlz2"
NOBJ = 2
K = 10
NDIM = NOBJ + K - 1
P = 30
H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))
BOUND_LOW, BOUND_UP = 0.0, 1.0
problem = pymop.factory.get_problem(PROBLEM, n_var=NDIM, n_obj=NOBJ)
##

# Algorithm parameters
MU = int(H + (4 - H % 4))
NGEN = 20
CXPB = 1.0
MUTPB = 1.0
seed = None
##

# Create uniform reference point
ref_points = tools.uniform_reference_points(NOBJ, P)

# Create classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)
creator.create("Individual", list, fitness=creator.FitnessMin, rank=None, nich=None, nich_dist=None, stress=None)


# Toolbox initialization
def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

toolbox = base.Toolbox()
toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", problem.evaluate, return_values_of=["F"])
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=30.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
##


def main(seed=None):
    random.seed(seed)

    # Initialize statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "HV", "std", "min", "avg", "max"
    logfile = open("logbook1_stats.log","w+")

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), HV="None     ", **record)
    logfile.write("{}\n".format(logbook.stream))

    # Begin the generational process
    for gen in range(1, NGEN):

        # If UNSGA3 == True then we apply the UNSGA3
        if UNSGA3 == True:
            Upop = tools.emo_mod.niching_selection_UNSGA3(pop)
            offspring = algorithms.varAnd(Upop, toolbox, CXPB, MUTPB)
        else:
            offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)   

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, MU)
        HV = hypervolume(pop, [1]*NOBJ)

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), HV=HV, **record)
        logfile.write("{}\n".format(logbook.stream))


    return pop



last_pop = main(seed=seed)


# Save the objective function values of the last (best) population
pop_fit = []
for ind, k in zip(last_pop, range(len(last_pop))):
    pop_fit.append(ind.fitness.values)

pop_fit = numpy.array(pop_fit) 
#print(pop_fit)

# Find best solution
best_idx=BestSol(pop_fit, weights=[1/NOBJ]*NOBJ, normalize=False).EUDIST()

print(best_idx)
print(pop_fit[best_idx])
print(numpy.shape(last_pop))


# Make a Plot
if NOBJ == 2:
    plot1 = ExaPlots.ObjFun2D(ref_points, pop_fit, best_idx)
elif NOBJ == 3:
    plot1 = ExaPlots.ObjFun3D(ref_points, pop_fit, best_idx)
else:
    pass


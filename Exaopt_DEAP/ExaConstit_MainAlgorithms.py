from deap import creator, base, tools, algorithms
import numpy
import random
from math import factorial
import pickle
import sys

from ExaConstit_Problems import ExaProb
from ExaConstit_SolPicker import BestSol



class ExaAlgorithms(ExaProb):

    def __init__(self,
                problem,
                seed,
                checkpoint,
                checkpoint_freq,    
                NOBJ,
                NPOP,
                NGEN,
                BOUND_LOW,
                BOUND_UP,
                **args):

        self.NOBJ = NOBJ
        self.NDIM = len(BOUND_LOW)
        self.BOUND_LOW = BOUND_LOW
        self.BOUND_UP = BOUND_UP
        self.problem = problem

        #====================== Initialize Optimization Strategy ======================
        # Create minimization problem (multiply -1 weights)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)
        # Create the Individual class that it has also the fitness (obj function results) as a list
        creator.create("Individual", list, fitness=creator.FitnessMin, stress=None)


    def NSGA3_main(self, ref_points, **args):



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
        # Register the above individual generator method in the toolbox class. That is attr_float with arguments low=self.BOUND_LOW, up=self.BOUND_UP, size=NDIM
        toolbox.register("attr_float", uniform, self.BOUND_LOW, self.BOUND_UP, self.NDIM)
        # Function that produces a complete individual with NDIM number of genes that have Low and Up bounds
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        # Function that instatly produces MU individuals (population). We assign the attribute number_of_population at the main function in this problem
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)


        #=========================== Initialize GA Operators ============================
        #### Evolution Methods
        # Function that returns the objective functions values as a dictionary (if n_obj=3 it will evaluate the obj function 3 times and will return 3 values (str) - It runs the problem.evaluate for n_obj times)
        toolbox.register("evaluate", self.problem.evaluate)   # Evaluate obj functions
        # Crossover function using the cxSimulatedBinaryBounded method
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=self.BOUND_LOW, up=self.BOUND_UP, eta=30.0)
        # Mutation function that mutates an individual using the mutPolynomialBounded method. A high eta will producea mutant resembling its parent, while a small eta will produce a ind_fitution much more different.
        toolbox.register("mutate", tools.mutPolynomialBounded, low=self.BOUND_LOW, up=self.BOUND_UP, eta=20.0, indpb=1.0/self.NDIM)
        # Selection function that selects individuals from population + offspring using selNSGA3 method (non-domination levels, etc (look at paper for NSGAIII))
        toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

        
        #================================ Evolution Algorithm ===========================
        # Start NSGA algorithm
        iter_tot, pop_fit, pop_param, pop_stress = main(self.seed, self.checkpoint, self.checkpoint_freq)

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
                    iter_tot = ckp["iter_tot"]
                    start_gen = ckp["generation"] + 1
                    if start_gen>NGEN: gen = start_gen
                    logbook1 = ckp["logbook1"]
                    logbook2 = ckp["logbook2"]
                except:
                    print("\nERROR: Wrong Checkpoint file")

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
                break_count = 1
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
                fitness_eval = toolbox.map(toolbox.evaluate, invalid_ind)
                

                # Evaluates the fitness for each invalid_ind and assigns them the new values
                for ind, fit in zip(invalid_ind, fitness_eval): 
                    iter_pgen+=1
                    iter_tot+=1
        #_______________________________________________________________________________________________
                    while self.problem.is_simulation_done() != 0 and break_count <= break_limit:

                        ind1 = invalid_ind[random.randrange(NPOP)]
                        ind2 = invalid_ind[random.randrange(NPOP)]
                        new_ind = toolbox.mate(ind1, ind2)[0]                   
                        new_ind = toolbox.mutate(new_ind)[0]
                        
                        text="Attempt to find another Parameter set to converge, break_count = {}\n\n".format(break_count)
                        self.problem.write_ExaProb_log(text, "warning", changeline=False)
                        fit = toolbox.evaluate(new_ind) 

                        break_count+=1
                        if break_count > break_limit: 
                            text = "The evaluation failed for a total of {} attempts! Framework will terminate!".format(break_count-1)
                            self.problem.write_ExaProb_log(text, "error", changeline=True)
                            sys.exit()
        #_______________________________________________________________________________________________
                    ind.fitness.values = fit
                    ind.stress = self.problem.return_stress()

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
                offspring = algorithms.varAnd(pop, toolbox, 1, 1)    
                        
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
                    if problem.is_simulation_done() != 0:
                        break_count+=1
                        new_ind = toolbox.mutate(invalid_ind[iter_pgen-1])
                        print("new failed ind = {}".format(new_ind))
                        fit = toolbox.evaluate(new_ind) 
                        print("new failed fit: {}".format(fit))
        #_______________________________________________________________________________________________
                    ind.fitness.values = fit
                    print("this normal fit: {}".format(fit))
                    ind.stress = problem.return_stress()

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
                    ckp = dict(population=pop, pop_fit = pop_fit, pop_param=pop_param, pop_stress=pop_stress, iter_tot=iter_tot, generation=gen, logbook1=logbook1, logbook2=logbook2, rndstate=random.getstate())
                    with open("checkpoint_files/checkpoint_gen_{}.pkl".format(gen), "wb+") as ckp_file:
                        pickle.dump(ckp, ckp_file)
                    # Fill the dictionary using the dict(key=value[, ...]) constructor
                    out = dict(pop_fit = pop_fit, pop_param=pop_param, pop_stress=pop_stress, iter_tot=iter_tot, generation=gen)
                    with open("checkpoint_files/output_gen_{}.pkl".format(gen), "wb+") as out_file:
                        pickle.dump(out, out_file)
            
            logfile1.close()
            logfile2.close()

            return iter_tot, pop_fit, pop_param, pop_stress


    def eval_fit_failure_handler(sefl, invalid_ind, break_limit, break_count):

        while self.problem.is_simulation_done() != 0:
            
            ind1 = invalid_ind[random.randrange(NPOP)]
            ind2 = invalid_ind[random.randrange(NPOP)]
            new_ind = toolbox.mate(ind1, ind2)[0]                   
            new_ind = toolbox.mutate(new_ind)[0]

            text="Attempt to find another Parameter set to converge, break_count = {}\n\n".format(break_count)
            self.problem.write_ExaProb_log(text, "warning", changeline=False)
            fit = toolbox.evaluate(new_ind) 

            if not break_count <= break_limit: 
                text = "The evaluation failed for a total of {} attempts! Framework will terminate!".format(break_count-1)
                problem.write_ExaProb_log(text, "error", changeline=True)
                sys.exit()
            
            break_count+=1
        
        return fit
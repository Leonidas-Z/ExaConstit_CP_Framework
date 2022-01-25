# -*- coding: utf-8 -*-

import numpy as np
import os
import os.path
import subprocess
import sys
import math
import logging
import pygad
import Matgen_GA


# ========================================= GENERAL INPUTS =============================================
## Use the following line if we do not want same random numbers (Ask Gumpy developers for that)
'''
    # restart MATLAB and generate a new seed
    rng('shuffle')
'''
## File names
# Name of the options toml
options_toml = 'mtsdd_bcc.toml'
# File name of the macroscopic stress-strain simulation output
avg_stress_file_name = 'test_mtsdd_bcc_stress.txt'
# File name of the Experimental stress strain data with same stress units as in simulation
avg_exp_file_name = 'Experiment_strain_stress.txt'
# Make log file to track the runs. This file will be created after the code starts to run.
logging.basicConfig(filename='CPFit_GA.log', level = logging.INFO, format='%(message)s', datefmt='%m/%d/%Y %H:%M:%S ', filemode='w')
logger = logging.getLogger()

#======================================== ExaConstit Options ===========================================
# Number of cpus
ncpus = 4


# ========================================== READ FILES ================================================
# Read experimental stress data
S_exp_data = np.loadtxt(avg_exp_file_name, dtype='float') 
S_exp = np.array( S_exp_data[:,1] )
# Number of experimental stress data available. (Need to change time-steps accordingly in .toml)
no_datapoints = np.size(S_exp) 


# =================================== DEFINE OBJECTIVE FUNCTION ========================================== 
iter = 0                              # Initialize iteration count
iter_max = 1                          # Initialize iteration corresponding to minimum objective function
S_gfit = np.zeros((no_datapoints,1))  # Initialize stresses corresponding to the minimum objective function value

# Objfun function returns the objective function value.
def Objfun(x, solution_idx):  # f_max, x_max are not used currently
    global iter, S_exp, gen_count, f_max, x_max, iter_max, S_gfit, ncpus, no_datapoints, options_toml, avg_stress_file_name
    
    # To know the number of the iteration
    iter = iter+1

    # Create the logger file
    logger = logging.getLogger()
    logger.info('INFO: Iteration: %d', iter)  # Show iter in log file
    logger.info("Solution: x = "+str(x))  # Keep x parameters in log file

    # Delete file: stress_strain.txt
    if os.path.exists(avg_stress_file_name):
        os.remove(avg_stress_file_name)

    # Create mat file: props_cp_mts.txt
    Matgen_GA.Matgen(x)

    # Call ExaConstit to run the CP simulation
    logger.info('Waiting ExaConstit simulation to finish...')
    init_spack = '. ~/spack/share/spack/setup-env.sh && spack load mpich@3.3.2'
    run_exaconstit = 'mpirun -np %d ~/exaconstit_installation/ExaConstit/build/bin/mechanics -opt ./%s' % (ncpus, options_toml)
    status = subprocess.call(init_spack+' && '+run_exaconstit, shell=True)

    # Check if simulation is finished (flag = 0 -> successful)
    if os.path.exists(avg_stress_file_name):

        # Read the stress data 
        S_sim_data = np.loadtxt(avg_stress_file_name, dtype='float')

        # The macroscopic stress in the direction of load is the 3rd column (z axis)
        # We use unique so to exclude repeated values from cyclic loading steps. Is it relevent for ExaConstit?
        if np.ndim(S_sim_data) > 1:
            S_sim = np.unique(np.array(S_sim_data[:, 2]))
        else:
            S_sim = np.array(S_sim_data[2])
        # Final size of S_sim
        S_sim_size = np.size(S_sim)

        # Check if output #rows is the same as time steps (so that all time steps were successful in the simulation process)
        if status == 0 and S_sim_size == no_datapoints:
            flag = 0  # simulation is successful
            logger.info('SUCCESSFUL SIMULATION, flag = %d' % (flag))
        # To mitigate unconverged results we will make S_exp same size as S_sim
        elif S_sim_size < no_datapoints and S_sim_size > 5:
            flag = 1  # partially successful
            logger.warning('WARNING: Simulation has unconverged results. \nOutput data are not the same as the time steps for iteration = %d' % (iter))
            logger.warning('The size of the output file row is %d when time steps are %d' % (S_sim_size, no_datapoints))
            logger.warning('flag = %d' % (flag))
            S_exp = S_exp[0:S_sim_size]
        elif S_sim_size < 5:
            flag = 2
            logger.error('\nERROR: The simulation retrived less than 5 data. S_sim_size = %d'% (S_sim_size))
            logger.error('ERROR: flag = %d' % (flag))
            sys.exit('ERROR: PLEASE LOOK AT LOG FILE FOR MORE INFO')
        elif S_sim_size > no_datapoints: 
            flag = 3
            logger.error('\nERROR: Please use same or less time steps in simulation as the number of experimental data')
            logger.error('ERRRO: flag = %d' % (flag))
            logger.error('The size of the output file row is %d when time steps are %d' % (S_sim_size, no_datapoints))
            sys.exit()
        # Check if the simulation data has strain at these increments using the same strain increments (one more error) 
    


        ## Evaluate the objective function (sqrt of sum of squared differences b\w sim and expt 1st Moment)
        ## Keep the S_sim data in the log file and print f
        f = -math.sqrt(sum((S_exp - S_sim)**2)/sum(S_exp)**2)
        logger.info('Objective function result: f = %.11f' % (f))
        logger.debug('Macroscopic stress simulation output: \n\t'+str(S_sim))

        # Keep track of the f min value
        # write out after the of each generation
        if f >= f_max:
            f_max = f
            x_max = x
            iter_max = iter
            S_gfit = S_sim
            logger.debug('\nBest f so far for iteration = %d with CP: x = ' % (iter_max)+str(x_max)+' and min objective value: f = %f' % (f_max))
            logger.debug('Macroscopic stress that gives f_max: \n\t'+str(S_gfit))

        if (iter - (2*sol_per_pop - num_parents_mating)) % (sol_per_pop - num_parents_mating) == 0 and not(np.any(iter==np.arange(1, sol_per_pop))):
            gen_count = gen_count + 1

        logger.info('\n')

        return f

    else:
        flag = 4  # simulation failed
        logger.error('\nERROR: SIMULATION FAILED!!!')
        logger.error('ERROR: flag = %d' % (flag))
        logger.error('ERROR: Output file was not generated at iteration = %d' % (iter))
        sys.exit()


##  ==================================== GENETIC ALGORITHM INPUTS ======================================
f_max = -1000000  # Initialize maximum objective function value 
gen_count = -1    # Initialize ounter for number of generations completed

# Number of generations
num_generations = 5
# Number of solutions to be selected as parents in the mating pool.
num_parents_mating = 3 # The number of parents that are mating, will give birth to sol_per_pop - num_parents_mating = children_number. Thus will do children_number iterations after the 0 generation which creates the initial_population = sol_per_pop.
# Number of solutions in the population (people).
sol_per_pop = 4
# Total iterations
total_iterations = (sol_per_pop - num_parents_mating) * (num_generations + 1) + sol_per_pop
# Number of the parameters
num_parameters_genes = 4
# Type of each parameter
gene_type = [int, [float,6], int, [float,6]]
# Set the lower and upper bounds of the CP parameters x
xlim0 = list(np.arange(1500, 2500, 1))
xlim1 = {'low': 1.0e-4, 'high': 10e-4}            # {'low': 1, 'high': 5} floating-point value from the range that starts from 1 (inclusive) and ends at 5 (exclusive).
xlim2 = list(np.arange(280, 320, 1))              # From 280 to 320 need to write: list(range(280,321)) or list(np.arange(280, 320, 1, dtype=int))
xlim3 = {'low': 1e-2, 'high': 5.0e-2}             # list(np.linspace(1e-2, 5.0e-2, 100))
gene_space = [xlim0, xlim1, xlim2, xlim3]
# Initial population
# initial_population = [np.array([x0]*sol_per_pop)]


# ================================= GENETIC ALGORITHM OPTIMIZATION ========================================
# Genetic Algorithm (https://pygad.readthedocs.io/en/latest/) 
def on_generation(ga_instance):
    global last_fitness
    logger.info("======================= End of a Generation =======================")
    #logger.info("Generation number leo = %d" % (gen_count))
    logger.info("Generation number = {generation}".format(generation=ga_instance.generations_completed))
    logger.info("\nSo far:\niter_max = " + str(iter_max))
    logger.info("x_max = " + str(x_max))
    #logger.info("f_max leo = %.11f" % (f_max))
    logger.info("f_max = {fitness}".format(fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
    logger.info("===================================================================\n\n")
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

ga_instance = pygad.GA(fitness_func = Objfun,
                       num_generations = num_generations,
                       num_parents_mating = num_parents_mating,
                       sol_per_pop = sol_per_pop,
                       num_genes = num_parameters_genes,
                       parent_selection_type="sss",
                       gene_space = gene_space,
                       gene_type = gene_type,
                       mutation_percent_genes=0.01,
                       mutation_type="random",
                       mutation_num_genes=1,
                       on_generation=on_generation,
                       crossover_type="uniform",
                       save_solutions=True)                         # !!!!!!!! save_solutions needed for graphs. Be careful for memory overflow !!!!!!!
                       #initial_population=initial_population,
                       #mutation_probability=None
                       #mutation_by_replacement=True,
                       #crossover_probability=None,
                       #stop_criteria=["reach_-0.004", "saturate_15"])


# Running the GA to optimize the parameters of the function and plot results
ga_instance.run()
x_solution, fit_solution, solution_idx = ga_instance.best_solution()
logger.info("*******************************************************************")
logger.info("\tParameters of the best solution : {solution}".format(solution = x_solution))
logger.info("\tFitness value of the best solution : {solution_fitness}".format(solution_fitness=fit_solution))
logger.info("\tBest fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))
logger.info("\tBest fitness value reached after " + str(iter_max)+' iterations')
logger.info('\tTotal ExaConstit runs (iterations) = '+str(total_iterations)+' k = '+str(iter))
logger.info("*******************************************************************")


# Multiple Plots - PyGAD, GA algorithm
ga_instance.plot_fitness(title="PyGAD - Generation vs. Fitness", linewidth=5)
ga_instance.plot_new_solution_rate(title="PyGAD - Generation vs. New Solution Rate", linewidth=5)
ga_instance.plot_genes(graph_type="plot",plot_type="plot",solutions="all")
ga_instance.plot_genes(graph_type="boxplot",solutions='all')





'''
def on_generation(ga_instance):
    global last_fitness
    logger.info("Generation = {generation}".format(generation=ga_instance.generations_completed))
    logger.info("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]


                       (num_generations, 
                       num_parents_mating, 
                       fitness_func, 
                       initial_population=None, 
                       sol_per_pop=None, 
                       num_genes=None, 
                       init_range_low=-4, 
                       init_range_high=4, 
                       gene_type=float, 
                       parent_selection_type="sss",                 #sss (steady-state selection), rws (roulete), sus (universal), rank (rank selection), random (random selection), tournament
                       keep_parents=-1,                             #ellitism
                       K_tournament=3, 
                       crossover_type="single_point", 
                       crossover_probability=None, 
                       mutation_type="random", 
                       mutation_probability=None, 
                       mutation_by_replacement=False, 
                       mutation_percent_genes='default', 
                       mutation_num_genes=None, 
                       random_mutation_min_val=-1, 
                       random_mutation_max_val=1, 
                       gene_space=None, 
                       allow_duplicate_genes=True, 
                       on_start=None, 
                       on_fitness=None,
                       on_parents=None,
                       on_crossover=None, 
                       on_mutation=None, 
                       callback_generation=None, 
                       on_generation=None, 
                       on_stop=None, 
                       delay_after_gen=0, 
                       save_best_solutions=False, 
                       save_solutions=False, 
                       suppress_warnings=False, 
                       stop_criteria=None)


## Original
options = gaoptimset('InitialPopulation',x0,'PopulationSize',popul_size,'EliteCount',8,...
    'CrossoverFraction',0.25,'Generations',no_gens,'FitnessLimit',1,'PlotFcns',{@gaplotbestf})

# Run genetic algorithm fitting (Check Matlab documentation to learn about
# input arguments)
[x_best,fval]  =ga(fitnessfcn,nvars,[],[],[],[],lb,ub,[],[2,4,5,7,8],options) #this calls the Objfun for many times #2 4 5 7 8 keep as integers
# Why not local 

fprintf(fid,'#s\n','End of simulation')


## Original
options = gaoptimset('InitialPopulation',x0,'PopulationSize',popul_size,'EliteCount',8,...
    'CrossoverFraction',0.25,'Generations',no_gens,'FitnessLimit',1,'PlotFcns',{@gaplotbestf})

# Run genetic algorithm fitting (Check Matlab documentation to learn about
# input arguments)
[x_best,fval]  =ga(fitnessfcn,nvars,[],[],[],[],lb,ub,[],[2,4,5,7,8],options) # keep as integers the flowing parameters: 2 4 5 7 8
'''
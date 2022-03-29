import numpy as np
import os
import os.path
import subprocess
import logging
import sys
from ExaConstit_MatGen import Matgen


class ExaProb:

    '''
    This is the constructor of the objective function evaluation
    All the assigned files must have same length with the n_obj
    (for each obj function we need a different Experiment data set etc.)
    for loc_file and loc_mechanics, give absolute paths
    if mult_GA=False, (run problem with simple GA), then n_obj argument doesn't do anything
    '''

    def __init__(self,
                 n_obj=2,
                 mult_GA=False,
                 n_steps=[20,20],
                 n_dep=None,
                 ncpus=2,
                 loc_mechanics="./mechanics",
                 #loc_input_files = "",
                 #loc_output_files ="",
                 Exper_input_files=['Experiment_stress_270.txt', 'Experiment_stress_300.txt'],
                 Sim_output_files=['test_mtsdd_bcc_stress.txt','test_mtsdd_bcc_stress.txt'],
                 Toml_files=['./mtsdd_bcc.toml', './mtsdd_bcc.toml'],
                 ):

        self.n_obj = n_obj
        self.n_dep = n_dep
        self.ncpus = ncpus
        # self.loc_input_files=loc_input_files
        # self.loc_output_files=loc_output_files
        self.loc_mechanics = loc_mechanics
        self.Toml_files = Toml_files
        self.Sim_output_files = Sim_output_files
        self.Exper_input_files = Exper_input_files
        self.iter = 0
        self.runs = 0
        self.mult_GA = mult_GA

        # Make log file to track the runs. This file will be created after the code starts to run.
        logging.basicConfig(filename='logbook3_ExaProb.log', level=logging.INFO,format='%(message)s', datefmt='%m/%d/%Y %H:%M:%S ', filemode='w')
        self.logger = logging.getLogger()

        # Check if we have as many files as the objective functions
        for data, name in zip([n_steps, Toml_files, Exper_input_files, Sim_output_files], ["n_steps", "Toml_files", "Exper_input_files", " Sim_output_files"]):
            if len(data) != n_obj and mult_GA==True:
                self.logger.error('ERROR: The length of "{}" is not equal to NOBJ={}'.format(name, n_obj))
                sys.exit('\nERROR: The length of "{}" is not equal to NOBJ={}'.format(name, n_obj))
            if mult_GA==False:
                self.logger.error('ERROR: The length of "{}" is not equal to len(Exper_input_files)={}'.format(name, len(Exper_input_files)))
                sys.exit('\nERROR: The length of "{}" is not equal to len(Exper_input_files)={}'.format(name, len(Exper_input_files)))
        
        # if multi-objective scheme, we should have more that 2 objectives
        if n_obj<2 and mult_GA==True:
            self.logger.error('ERROR: NOBJ={} when mult_obj=True'.format(n_obj))
            sys.exit('\nERROR: NOBJ={} when mult_obj=True'.format(n_obj))

        # Read Experiment data sets and save to S_exp
        # Check if the length of the S_exp is the same with the assigned n_steps in the toml file
        self.S_exp = []
        for k in range(len(Exper_input_files)):
            try:
                S_exp_data = np.loadtxt(Exper_input_files[k], dtype='float', ndmin=2)
            except:
                self.logger.error("ERROR: Exper_input_files[{k}] was not found!".format(k=k))
                sys.exit("ERROR: Exper_input_files[{k}] was not found!".format(k=k))

            # Assuming that each experment data file has only a stress column
            # S_exp will be a list that contains a numpy array corresponding to each file
            S_exp = S_exp_data[:, 0]
            self.S_exp.append(S_exp)

            if n_steps[k] != len(S_exp):
                self.logger.error("ERROR: The length of S_exp[{k}] is not equal to n_steps[{k}]".format(k=k))
                sys.exit("\nERROR: The length of S_exp[{k}] is not equal to n_steps[{k}]".format(k=k))



    def evaluate(self, x):
        # Iteration and logger info down below modifies class
        # In a parallel environment if we want the same behavior as down
        # below, we will need some sort of lock associated with things.
        # We would need to lock around the logger and self.iter
        #
        # Alternatively, we would want to have things associated with the gene # we're modifying
        # and the global iteration we're on.
        # Log files would need to be per process as well if we did the above
        # Alternatively, we could collect all this various information and within a vector\list
        # of length # genes and store the logging information in that list.
        # We could do the same for objective function fits as well. Once, a given iteration is
        # finished we could do a mpi_send of the data to rank 0 which could then save everything
        # off to a single file... (not sure if this is possible within DEAP)

        # Separate parameters into dependent (thermal) and independent (athermal) groups
        # x_group[0] will be the independent group. The rest will be the dependent groups for each objective
        if self.n_dep != None:
            self.n_ind = len(x) - self.n_obj*self.n_dep
            x_dep = x[self.n_ind:]
            x_group = [x[0:self.n_ind]]
            x_group.extend([x_dep[k:k+self.n_dep] for k in range(0, len(x_dep), self.n_dep)])
        else:
            self.n_ind = len(x)
            x_group = [x[0:self.n_ind]]

        # Count iterations and save solutions
        self.iter += 1
        self.logger.info("INFO: Iteration: {}".format(self.iter))
        self.logger.info("\t\tSolution: x = {}".format(x_group))

        # Initialize
        self.S_sim = []
        f = np.zeros(self.n_obj)

        # Run k simulations. One for each objective function
        for k in range(self.n_obj):
            # Within this loop we could automatically generate the option file and job directory
            # We can then within here cd to the subdirectory that we generated
            # Count GA and Exaconstit iterations
            self.runs += 1
            self.logger.debug('\tExaConstit Runs: %d' % self.runs)

            # Delete file contents of the Simulation output file
            if os.path.exists(self.Sim_output_files[k]):
                file = open(self.Sim_output_files[k], 'r+')
                file.truncate(0)
                file.close()

            # Create mat file: props_cp_mts.txt and use the file for multiobj if more files
            try:
                if self.n_dep != None:
                    Matgen(x_ind=x_group[0], x_dep=x_group[k+1])
                else:
                    Matgen(x_ind=x_group[0])
            except:
                self.logger.error("ERROR: Unable to generate material properties using Matgen!")
                sys.exit("\nERROR: Unable to generate material properties using Matgen!")

            # Call ExaConstit to run the CP simulation
            self.logger.info('\tWaiting ExaConstit for file %s ......'% self.Exper_data_files[k])
            # spack loading portion wasn't needed do this outside the python script and then you're good to go
            run_exaconstit = 'mpirun -np {ncpus} {mechanics} -opt {toml_name}'.format(ncpus=self.ncpus, mechanics=self.loc_mechanics, toml_name=self.Toml_files[k])
            # quite all the stdout from this. We could have this saved off to a logger potentially or to a given output file
            # if that is a desired behavior
            status = subprocess.call(run_exaconstit, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDERR)

            # Read the simulation output
            # If output file exists and it is not empty, read stress
            if os.path.exists(self.Sim_output_files[k]) and os.stat(self.Sim_output_files[k]).st_size != 0:
                S_sim_data = np.loadtxt(self.Sim_output_files[k], dtype='float', ndmin=2)
                # Macroscopic stress in the direction of load: 3rd column (z axis)
                _S_sim = S_sim_data[:, 2]
                # We use unique so to exclude repeated values from cyclic loading steps. Is it relevent for ExaConstit?
                S_sim = np.unique(_S_sim)
            else:
                self.logger.error('\nERROR: Simulation did not run for iteration = {}, Exper_input_files[{}]'.format(self.iter, k))
                sys.exit('\nERROR: Simulation did not run for iteration = {}, Exper_input_files[{}]'.format(self.iter, k))

            # Need more thought!!!!!!!!
            # Check if data size is the same with experiment data-set in case there is a convergence issue
            no_sim_data = len(S_sim)
            no_exp_data = len(self.S_exp[k])
            if status == 0 and no_sim_data == no_exp_data:
                self.logger.info('\t\tSUCCESSFULL SIMULATION!!!')
            else:
                self.logger.info('\nERROR: SIMULATION FAILED!\nERROR: Sim_output_files[{k}] does not have same raw dimension as Exper_input_files[{k}]: no_sim_data={s}, no_exp_data={e}'.format(k=k, s=no_sim_data, e=no_exp_data))
                sys.exit('\nERROR: SIMULATION FAILED!\nERROR: Sim_output_files[{k}] does not have same raw dimension as Exper_input_files[{k}]: no_sim_data={s}, no_exp_data={e}'.format(k=k, s=no_sim_data, e=no_exp_data))

            # S_sim will be a list that contains a numpy array of stress corresponding to each file
            self.S_sim.append(S_sim)

            # Evaluate the individual objective function. Will have k functions. (Normalized Root-mean-square deviation (RMSD)- 1st Moment (it is the error percentage))
            # We take the absolute values to compensate for the fact that in cyclic simulations we will have negative and positive values
            S_exp_abs = abs(self.S_exp[k])
            S_sim_abs = abs(self.S_sim[k])
            
            f[k] = np.sqrt(sum((S_sim_abs-S_exp_abs)**2)/sum(S_exp_abs**2))
            self.logger.info('\t\tIndividual obj function: fit = '+str(f[k]))
               
        self.logger.info('')

        # If use a simple GA scheme then return the summation of all the objective functions
        # If use a multiple_objective GA scheme then return individual objective functions
        if self.mult_GA == False:
            F = sum(f)
            self.logger.info('\tGlobal obj function: fit = '+str(F))
        else:
            F = f

        return F

    def return_stress(self):
        # save stresses in a list for the particular iteration that returnStress() function is called
        stress = []
        stress.append(self.S_exp)
        stress.append(self.S_sim)
        #for k in range(len(self.Exper_input_files)):
            #stress.extend([[ self.S_exp[k], self.S_sim[k] ]])
        return stress

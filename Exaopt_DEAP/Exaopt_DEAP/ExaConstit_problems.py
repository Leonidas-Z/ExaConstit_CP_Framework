import numpy as np
import os
import os.path
import subprocess
import logging
import sys
import Matgen_GA_Multiobj



class ExaProb:
    # This is the constructor of the CP_Optimization class
    def __init__(self,
                 n_obj,
                 n_var,
                 #threshold,
                 x_dep=[],
                 x_indep=[],
                 ncpus = 4,
                 options_toml = 'mtsdd_bcc.toml',
                 Simul_data_file = 'test_mtsdd_bcc_stress.txt',
                 Exper_data_files = ['Experiment_stress_270.txt', 'Experiment_stress_300.txt']):
    
        self.n_var = n_var
        self.n_obj = n_obj
        self.x_dep = x_dep
        self.x_indep = x_indep
        #threshold
        self.ncpus = ncpus
        self.options_toml = options_toml
        self.Simul_data_file = Simul_data_file
        self.Exper_data_files = Exper_data_files 
        self.iter = 0
        self.runs = 0

        # Make log file to track the runs. This file will be created after the code starts to run.
        logging.basicConfig(filename='Logbook3_ExaConstit.log', level = logging.INFO, format='%(message)s', datefmt='%m/%d/%Y %H:%M:%S ', filemode='w')
        self.logger = logging.getLogger()

        if n_obj != len(Exper_data_files):
            self.logger.error('\nERROR: The number of the Experiment Data files is not the same with the number of the objective functions n_obj')
            sys.exit('\nERROR: The number of the Experiment Data files is not the same with the number of the objective functions n_obj') 
        
        no_exper_data = []
        S_exp = []
        for k in range(n_obj):
            try:
                S_exp_data = np.loadtxt(Exper_data_files[k], dtype='float')
                S_exp_stress = S_exp_data[:,1]
                S_exp.append(S_exp_stress)
                no_exper_data.append(len(S_exp_stress))
            except FileNotFoundError:
                self.logger.error("ERROR: THE FILE {} WAS NOT FOUND!!!".format(Exper_data_files[k]))
                sys.exit("\nERROR: THE FILE {} WAS NOT FOUND!!!".format(Exper_data_files[k]))

        self.no_exper_data=no_exper_data
        self.S_exp=S_exp


    # This is the obj_function Method
    def evaluate(self, x):
        
        #logger.info('INFO: Iteration: %d', iter)  
        self.iter += 1 
        self.logger.info('INFO: Iteration: %d', self.iter) 
        self.logger.info("\t\tSolution: x = "+str(x))  

        # Specify elements of x as x_indep and x_dep for all files
        x_indep = x[0:3]          # Specify indep. array positions. Here first 3
        x_dep = [x[3:6], x[6:]]   # Specify dep array postions for every k
            

        S_sim = []  
        no_sim_data = []
        fit_obj = np.zeros(self.n_obj)

        for k in range(self.n_obj):
            # Count GA and Exaconstit iterations
            self.runs += 1
            self.logger.debug('\tExaConstit Runs: %d'% self.runs)

            # Delete file contents of the Simulation output file
            if os.path.exists(self.Simul_data_file):
                file = open(self.Simul_data_file, 'r+')
                file.truncate(0)
                file.close

            # Create mat file: props_cp_mts.txt and use the file for multiobj if more files 
            if self.n_obj > 1:
                try:
                    Matgen_GA_Multiobj.Matgen(x_indep, x_dep[k])
                except FileNotFoundError:
                    self.logger.error("ERROR: No material properties file has been found!!!")
                    sys.exit('\nERROR: NO MATERIAL PROPERTIES FILE WAS FOUND')
            elif self.n_obj == 1:
                try:
                    Matgen_GA.Matgen(x)
                except FileNotFoundError:
                    self.logger.error("ERROR: No material properties file has been found!!!")
                    sys.exit('\nERROR: NO MATERIAL PROPERTIES FILE WAS FOUND')


            # Call ExaConstit to run the CP simulation
            self.logger.info('\tWaiting ExaConstit for file %s ......'% self.Exper_data_files[k])
            init_spack = '. ~/spack/share/spack/setup-env.sh && spack load mpich@3.3.2'
            run_exaconstit = 'mpirun -np %d ~/exaconstit_installation/ExaConstit/build/bin/mechanics -opt ./%s' % (self.ncpus, self.options_toml)
            status = subprocess.call(init_spack+' && '+run_exaconstit, shell=True)


            # Read the simulation output
            if os.path.exists(self.Simul_data_file):
                S_sim_data = np.loadtxt(self.Simul_data_file, dtype='float')
            else:
                self.logger.error('\nERROR: Output file was not generated at iteration = %d ' % (iter)+ 'at exaCosntit run k = %d' % (k))
                sys.exit('\nERROR: NO OUTPUT FILE FROM SIMULATION WAS GENERATED')


            # We use unique so to exclude repeated values from cyclic loading steps. Is it relevent for ExaConstit?
            if np.ndim(S_sim_data) > 1:
                S_sim_stress_Z = S_sim_data[:, 2]           # The macroscopic stress in the direction of load is the 3rd column (z axis)
                S_sim_stress_Z = np.unique(S_sim_stress_Z)
                S_sim.append(S_sim_stress_Z)                # Save S_sim[k]
            else:
                S_sim_stress_Z = S_sim_data[2]              # if 1D array
                S_sim.append(S_sim_stress_Z)                
            
            no_sim_data.append(len(S_sim_stress_Z))         # Save final size of S_sim[k] 

            ############## Need more thought!!!!!!!!
            # Check if data size is the same with experiment data-set
            if status == 0 and no_sim_data[k] == self.no_exper_data[k]:
                self.logger.info('\t\tSUCCESSFULL SIMULATION!!!')
            else:
                self.logger.info('\nERROR: The simulation for file k = %d different number of data points than the experimental data. no_sim_data = %d'% (k, no_sim_data[k]))
                sys.exit('\nERROR: PLEASE LOOK AT GA_LOG FILE FOR MORE INFO')
          

            ## Evaluate the individual objective function. Will have k functions. (sqrt of sum of squared differences b\w sim and exp, 1st Moment)
            S_exp = self.S_exp
            fit_obj[k] = np.sqrt((1/len(S_exp[k]))*sum((1 - S_sim[k]/S_exp[k])**2))   
            self.logger.info('\t\tIndividual obj function: fit_obj = '+str(fit_obj[k]))
        
        self.logger.info('')  # Leave a space line in the log file

        return fit_obj
import numpy as np
import os
import os.path
import subprocess
import logging
import sys
import Matgen_GA_Multiobj


class ExaProb:
    # This is the constructor of the CP_Optimization class
    # All the assigned files must have same length with the n_obj 
    # (for each obj function we need a different Experiment data set etc.)
    # for loc_file and loc_mechanics, give absolutet paths

    def __init__(self,
                 n_obj,
                 n_steps,
                 x_dep=[],
                 x_indep=[],
                 ncpus = 4,
                 loc_mechanics ="~/exaconstit_installation/ExaConstit/build/bin/mechanics",
                 #loc_input_files = "",
                 #loc_output_files ="",
                 Exper_data_files = ['Experiment_stress_270.txt', 'Experiment_stress_300.txt'],
                 Toml_files = ['./mtsdd_bcc.toml', './mtsdd_bcc.toml'],
                 Simul_data_files = ['test_mtsdd_bcc_stress.txt','test_mtsdd_bcc_stress.txt']):


        self.n_obj = n_obj
        self.x_dep = x_dep
        self.x_indep = x_indep
        self.ncpus = ncpus
        #self.loc_input_files=loc_input_files
        #self.loc_output_files=loc_output_files
        self.loc_mechanics=loc_mechanics
        self.Toml_files = Toml_files
        self.Simul_data_files = Simul_data_files
        self.Exper_data_files = Exper_data_files 
        self.iter = 0
        self.runs = 0

        # Make log file to track the runs. This file will be created after the code starts to run.
        logging.basicConfig(filename='logbook3_ExaProb.log', level = logging.INFO, format='%(message)s', datefmt='%m/%d/%Y %H:%M:%S ', filemode='w')
        self.logger = logging.getLogger()

        # Check if we have as many files as the objective functions
        for data, name in zip([n_steps, Exper_data_files, Toml_files, Simul_data_files],["n_steps", "Exper_data_files", "Toml_files", "Simul_data_files"]):
            if not len(data)==n_obj:
                self.logger.error('ERROR: The number of files assigned to "{}" is not equal to NOBJ={}'.format(name, n_obj))
                sys.exit('\nERROR: The number of files assigned to "{}" is not equal to NOBJ={}'.format(name, n_obj))

        # Read Experiment data sets and save to S_exp
        # Check if the length of the S_exp is the same with the assigned n_steps in the toml file
        no_exper_data = []
        S_exp = []
        for k in range(n_obj):
            try:
                S_exp_data = np.loadtxt(Exper_data_files[k], dtype='float')

                # 0 column is the stress
                S_exp_stress = S_exp_data#[:,0]   

                S_exp.append(S_exp_stress)
                no_exper_data.append(len(S_exp_stress))

                if not n_steps[k] == no_exper_data[k]:
                    self.logger.error("ERROR: The length of the S_exp[{k}] is not the same with the assigned n_steps[{k}]".format(k=k))
                    sys.exit("ERROR: The length of the S_exp[{k}] is not the same with the assigned n_steps[{k}]".format(k=k))
            
            except FileNotFoundError:
                self.logger.error("ERROR: THE FILE {} WAS NOT FOUND!!!".format(Exper_data_files[k]))
                sys.exit("\nERROR: THE FILE {} WAS NOT FOUND!!!".format(Exper_data_files[k]))

        self.no_exper_data=no_exper_data
        self.S_exp=S_exp



    def evaluate(self, x):
        
        #logger.info('INFO: Iteration: %d', iter)  
        self.iter += 1 
        self.logger.info('INFO: Iteration: %d', self.iter) 
        self.logger.info("\t\tSolution: x = "+str(x))  

        # Specify elements of x as x_indep and x_dep for all files
        x_indep = x[0:3]          # Specify indep. array positions. Here first 3
        x_dep = [x[3:6], x[6:]]   # Specify dep array postions for every k
        
        # Initialize
        S_sim = []  
        no_sim_data = []
        f = np.zeros(self.n_obj)

        # Run k simulations. One for each objective function
        for k in range(self.n_obj):
            # Count GA and Exaconstit iterations
            self.runs += 1
            self.logger.debug('\tExaConstit Runs: %d'% self.runs)

            # Delete file contents of the Simulation output file
            if os.path.exists(self.Simul_data_files[k]):
                file = open(self.Simul_data_files[k], 'r+')
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
                    pass
                    #Matgen_GA.Matgen(x)
                except FileNotFoundError:
                    self.logger.error("ERROR: No material properties file has been found!!!")
                    sys.exit('\nERROR: NO MATERIAL PROPERTIES FILE WAS FOUND')


            # Call ExaConstit to run the CP simulation
            self.logger.info('\tWaiting ExaConstit for file %s ......'% self.Exper_data_files[k])
            init_spack = '. ~/spack/share/spack/setup-env.sh && spack load mpich@3.3.2'
            run_exaconstit = 'mpirun -np {ncpus} {mechanics} -opt {toml_name}'.format(ncpus=self.ncpus, mechanics=self.loc_mechanics, toml_name=self.Toml_files[k])
            status = subprocess.call(init_spack+' && '+run_exaconstit, shell=True)


            # Read the simulation output
            if os.path.exists(self.Simul_data_files[k]):
                S_sim_data = np.loadtxt(self.Simul_data_files[k], dtype='float')
            else:
                self.logger.error('\nERROR: Output file was not generated at iteration = %d ' % (iter)+ 'at exaCosntit run k = %d' % (k))
                sys.exit('\nERROR: NO OUTPUT FILE FROM SIMULATION WAS GENERATED')

            # We use unique so to exclude repeated values from cyclic loading steps. Is it relevent for ExaConstit?
            if np.ndim(S_sim_data) > 1:
                S_sim_stress_Z = S_sim_data[:, 2]           # Need to have a nice message if error here - not obvious!!!!!! The macroscopic stress in the direction of load is the 3rd column in the stress output (z axis)
                S_sim_stress_Z = np.unique(S_sim_stress_Z)
                S_sim.append(S_sim_stress_Z)                # Save S_sim[k]
            else:
                S_sim_stress_Z = S_sim_data[2]              # if 1D array
                S_sim.append(S_sim_stress_Z)                
            
            no_sim_data.append(len(S_sim_stress_Z))         # Save final size of S_sim[k] 


            ############## Need more thought!!!!!!!!
            # Check if data size is the same with experiment data-set in case there is a convergence issue
            if status == 0 and no_sim_data[k] == self.no_exper_data[k]:
                self.logger.info('\t\tSUCCESSFULL SIMULATION!!!')
            else:
                self.logger.info('\nERROR: The simulation for file k = %d different number of data points than the experimental data. no_sim_data = %d'% (k, no_sim_data[k]))
                sys.exit('\nERROR: PLEASE LOOK AT GA_LOG FILE FOR MORE INFO')
          

            # Evaluate the individual objective function. Will have k functions. (Normalized Root-mean-square deviation (RMSD)- 1st Moment (it is the error percentage))
            # We take the absolute values to compensate for the fact that in cyclic simulations we will have negative and positive values
            S_exp = self.S_exp
            f[k] = np.sqrt(sum((1 - abs(S_sim[k]/S_exp[k]))**2)/len(S_exp[k]))   
            self.logger.info('\t\tIndividual obj function: fit = '+str(f[k]))
        
        self.logger.info('')

        return f

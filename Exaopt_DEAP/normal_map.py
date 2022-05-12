import os
import os.path
import subprocess
import sys
from ExaConstit_Logger import write_ExaProb_log

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def map_custom(problem, igeneration, genes):
    '''
    Probably won't be as efficient as just doing it the current map way
    but this should allow us to repeat this process more or less in other areas and
    have something that is like what the other mapping functions might do.
    '''

    status = []

    run_exaconstit = 'mpirun -np {ncpus} {mechanics} -opt {toml_name}'.format(ncpus=problem.ncpus, mechanics=problem.bin_mechanics, toml_name='options.toml')

    f_objective = []

    # Preprocess all of the genes first
    for igene, gene in enumerate(genes):
        problem.preprocess(gene, igeneration, igene)

    # Run all of the gene data next
    for igene, gene in enumerate(genes):
        istatus = []
        for iobj in range(problem.n_obj):
            rve_name = 'gen_' + str(igeneration) + '_gene_' + str(igene) + '_obj_' + str(iobj)
            fdironl = os.path.join(problem.workflow_dir, rve_name, "")
            # cd into directory and run command and then when this code block exits it returns us
            # to the working directory
            with cd(fdironl):
                istatus_o = subprocess.call(run_exaconstit, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDERR)
                istatus.append(istatus_o)
        status.append(istatus)

    # Post-process all of the data last
    for igene, gene in enumerate(genes):
        f = problem.postprocess(igeneration, igene, status[igene])
        f_objective.append(f)
    
    return f_objective

# Will want a custom way to handle one off launches for failed tests
        
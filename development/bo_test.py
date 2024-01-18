#!/usr/bin/env python3.10
#PBS -q short
#PBS -l nodes=1:ppn=10
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -N opt_bo
#PBS -o /beegfs/work/st_163811/vle_using_solvation_free_energy/development/LOG_BO
#PBS -l mem=1000mb

import os
import sys
import GPy
import logging
import numpy as np
import matplotlib.pyplot as plt

# Appending path to find utils 
sys.path.append("/beegfs/work/st_163811/vle_using_solvation_free_energy/development/") 
sys.path.append("/beegfs/work/st_163811/vle_using_solvation_free_energy/")

from GPyOpt.methods import BayesianOptimization

from scipy.interpolate import interp1d
from tools.gaussian_process import GPR
from tools.reader import get_dh_dl, get_data

from utils_automated import  train_gpr, get_partial_uncertanty


root = "/home/st/st_st/st_st163811/workspace/vle_using_solvation_free_energy/development/"

# Delete old logger file if exists
if os.path.exists(root+'bo.log'): os.remove(root+'bo.log')

# Create a logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)

# Create a file handler and set the level to debug
file_handler = logging.FileHandler(root+'bo.log')
file_handler.setLevel(logging.INFO)

# Create a formatter and set the format for log messages
formatter = logging.Formatter('%(asctime)s -  %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)


# Gather simulation results

data_output = "/home/st/st_st/st_st163811/workspace/vle_using_solvation_free_energy/simulation_systems/mixture_hexane_butylamine/373/hexane_coupled/x1.0/vdw/TI/sim_%d/fep%d%d.sampling"

idx_sim_folder = np.arange(0,81,1)
paths     = [data_output%(i,i,i+1) for i in idx_sim_folder]
lambdas   = np.round( np.concatenate( (np.linspace(0,0.2,60), np.linspace(0.22,1.0,21)) ), 3 )
mean, var = get_dh_dl( fe_data = [get_data(paths)], no_intermediates = len(lambdas), delta = 0.0001 , both_ways = False)

precision = 2
std_dG_opt = 0.011

gpr_modeling = train_gpr( lambdas, var, kernel = GPy.kern.Matern32 )

logger.info("Sucessfully trained GPR\n Start Bayesian optimization\n")

# Define your objective function
def objective_function(x):
    
    x = np.concatenate( [ np.zeros(1), x.flatten(), np.ones(1) ] )

    if np.unique( np.round( x, precision ) ).size < 10:
        rmsd = 1e6
    else:
        x = np.round( x, precision ).flatten()

        var_init = [ abs( gpr_modeling.predict([[xx]])[0].item() ) for xx in x ]
        
        std_dG_i = get_partial_uncertanty( x, var_init )

        # Relative root mean square deviation of the current iteration lambdas to the optimal value 
        # (which is computed as the mean of the input data and the number of newly intermediates)
        rmsd     = np.sqrt( np.mean( (np.array(std_dG_i) - std_dG_opt)**2 ) ) / std_dG_opt
    
    return rmsd

# Define the bounds of the input space
domain = [{'name': f'var_{i}', 'type': 'continuous', 'domain': (0.01, 0.99)} for i in range(8)]

# Create the Bayesian Optimization object
bayesian_optimizer = BayesianOptimization(f=objective_function, domain=domain, model_type='GP', num_cores=10)

# Run the optimization loop
max_iter = 500
bayesian_optimizer.run_optimization(max_iter)

# Get the optimized point and value
best_point = np.sort( np.round( np.concatenate( [ np.zeros(1), bayesian_optimizer.x_opt.flatten(), np.ones(1) ] ), precision ) )
best_value = np.round( bayesian_optimizer.fx_opt * 100 )

logger.info("Best point:\n" + " ".join([str(np.round(bp,3)) for bp in best_point]) + "\n")
logger.info(f"Best value: {best_value} \n")
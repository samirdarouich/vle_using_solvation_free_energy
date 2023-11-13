#!/usr/bin/env python3.10
#PBS -q short
#PBS -l nodes=1:ppn=1
#PBS -l walltime=15:00:00
#PBS -j oe
#PBS -N opt_intermediate_python
#PBS -o /beegfs/work/st_163811/vle_using_solvation_free_energy/development/lammps/ethanol_water/automated/LOG_python
#PBS -l mem=30mb

import subprocess
import os
import logging
import numpy as np
import sys 
import shutil

# Appending current path to find utils 

root = "/beegfs/work/st_163811/vle_using_solvation_free_energy/development/lammps/ethanol_water/automated/"
#root = sys.argv[0]
sys.path.append(root) 

from utils_automated import change_inputfile, change_jobfile, trackJobs, get_mean_sim_data_lo, train_gpr, get_new_lambdas

## Define logger ## 

# Create a logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)

# Create a file handler and set the level to debug
file_handler = logging.FileHandler(root+'opt_intermediates.log')
file_handler.setLevel(logging.INFO)

# Create a formatter and set the format for log messages
formatter = logging.Formatter('%(asctime)s -  %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

#### Python code that optimize intermediates for free solvation energy calculations ####

## Initial setup ##

# Define minimum number of simulations
N_min  = 10

# Define after which rmsd convergence is achieved
tol    = 0.1

# Start with an initial simulation of the system (these are equilibrated for 1.5ns and production of 0.5ns to sample)
l_init = [ 0.0, 0.5, 1.0]

# Define original input file for simulation setup

orig_file      = root+"../ethanol_TI_vdw.input"
file_output    = root+"sim_lj_%d/ethanol.input"
job_file       = root+"run.sh"
job_output     = root+"run_%d.sh"
data_output    = root+"sim_lj_%d/fep_lj.fep"
idx_sim_folder = []

# Setup original simulation folder
for i,li in enumerate(l_init):
    logger.info("Creating job files:\n %s"%(file_output%i))
    change_inputfile(orig_file,file_output%i,li,restart=False)
    idx_sim_folder.append(i)

# Submit simulations
job_list = []
for i in idx_sim_folder:
    # Write job file
    change_jobfile(job_file,job_output%i,"sim_lj_%d"%i)
    
    # Submit job file
    #exe = subprocess.run(["qsub",job_output%i],capture_output=True,text=True)
    #job_list.append( exe.stdout.split("\n")[0] )

logger.info("These are the submitted jobs:",job_list)

# Let python wait for the jobs to be finished (check job status every 5 min and if all jobs are done
# didnt check why they are done, then continue with this script
trackJobs(job_list)

## Start iterating that at least N_min intermediates are taken and the rmsd is lower than the tolerance ##

# Define learning input for GPR 
l_learn   = np.array([[]])
var_learn = np.array([[]])

# Define starting paramters for iteration
N_sim = len(l_init)
rmsd  = 1.0
itter = 0

while N_sim < N_min and rmsd > tol:

    N_sim = len(l_init)

    logger.info("\nIteration %d with current n° of intermediates: %d and a RMSD of %.1f%%\n"%(itter,N_sim,rmsd*100))

    # Gather simulation results
    paths     = [data_output%i for i in idx_sim_folder]
    mean, var = get_mean_sim_data_lo(paths,N_sim)
    
    logger.info("Current simulation results:",mean)

    if l_learn.size == 0:
        l_learn   = np.array(l_init).reshape(-1, 1)
        var_learn = np.array(var).reshape(-1, 1)
    else:
        l_learn   = np.concatenate( [ l_learn, np.array(l_init).reshape(-1, 1) ] )
        var_learn = np.concatenate( [ var_learn, np.array(var).reshape(-1, 1) ] )

    # Sort training data and make it unique
    _ ,idx       = np.unique( l_learn, return_index=True )
    l_learn      = l_learn[idx]
    var_learn    = var_learn[idx]

    # Train GPR
    gpr_modeling = train_gpr(l_learn,var_learn)

    # Get new lambdas (this means redistribute the current lambdas and add one point more) and the corresponding root mean square deviation (rmsd)
    # with BO or interpolation redistribution
    l_init, rmsd   = get_new_lambdas(l_init,var,gpr_modeling,method="BO",verbose=True)

    # Loop through new lambdas and check if simulation already exists and if not then start it
    idx_sim_folder = []
    idx_old_folder = []
    dummy          = np.unique( np.concatenate( [ l_learn, np.array(l_init).reshape(-1, 1) ] ) )
    
    # Rename existing sim folders that they match the new lambda order and indexes
    sim_folder_names = os.path.dirname(data_output)

    # Loop recoursevly through the lambdas to rename folders without overwriting
    for nli,nl in zip( np.arange(len(dummy)-1,-1,-1), np.flip(dummy) ):
        # If lambda already simulated rename the old lambda index (oli) folder to match the new lambda index (nli) folder
        if nl in l_learn:
            oli = np.where( l_learn.flatten() == nl )[0][0]
            shutil.move( sim_folder_names%oli, sim_folder_names%nli )
            #print("old lambda: %.2f in folder %s will be renamed to %s"%(nl,sim_folder_names%oli, sim_folder_names%nli))
        
        # If lambda was not simulated bevore
        else:
            os.mkdir( sim_folder_names%nli )
            #print("new lambda: %.2f in folder %s"%(nl,sim_folder_names%nli))
            

    for l in l_init:
        # Get difference to all simulated lambdas
        diff = [ np.abs(l - ll) for ll in l_learn.flatten() ] 

        # If point already exist dont redo simulation #
        if np.min(diff) == 0.0:
            # Get this new (old) lambda point and add its index to new sim folders that 
            # should be evaluated next iteration. Dummy contains the indexes after adding the new lambdas
            new_old_lambda = l_learn.flatten()[ np.argmin(diff) ]
            idx_old_folder.append( np.where( dummy == new_old_lambda)[0][0] )

            #print("\nlambda %.2f exists"%l)
            #print("idx: %d\n"%np.where( dummy == new_old_lambda)[0][0])
        
        # If the point needs to be simulated
        else:
            # Search at which index the new lambda is added in the sorted list (dummy)
            idx                = np.where( dummy == l)[0][0]
            nearest_lambda     = l_learn.flatten()[ np.argmin(diff) ]
            idx_nearest_lambda = np.where( dummy == nearest_lambda)[0][0]
            
            # Create new input file in the new folder with the restart information of the closest system
            change_inputfile(orig_file,file_output%idx,l,restart=True,
                             restart_file="../sim_lj_%d/equil.restart"%idx_nearest_lambda)
            idx_sim_folder.append( idx )

            #print("\nlambda %.2f does not exist, create new one"%l)
            #print("using %.2f as nearest restart. with sim_folder %d\n"%(nearest_lambda,idx_nearest_lambda))

    # Submit simulations
    job_list = []
    for i in idx_sim_folder:
        # Write job file
        change_jobfile(job_file,job_output%i,"sim_lj_%d"%i)

        # Submit job file
        exe = subprocess.run(["qsub",job_output%i],capture_output=True,text=True)
        job_list.append( exe.stdout.split("\n")[0] )

    # Let python wait for the jobs to be finished (check job status every 5 min and if all jobs are done
    # didnt check why they are done, then continue with this script
    trackJobs(job_list)
    
    # merge already simulated and new simulated idx list to evaluate correctly in new iteration 
    idx_sim_folder += idx_old_folder
    idx_sim_folder.sort()



    
#!/usr/bin/env python3.10
#PBS -q long
#PBS -l nodes=1:ppn=1
#PBS -l walltime=96:00:00
#PBS -j oe
#PBS -N opt_intermediate_python
#PBS -o /beegfs/work/st_163811/vle_using_solvation_free_energy/development/lammps/butylamine_butylamine/adapted/LOG_python
#PBS -l mem=1000mb

import subprocess
import os
import logging
import numpy as np
import sys 
import shutil
import matplotlib.pyplot as plt

# Appending path to find utils 
sys.path.append("/beegfs/work/st_163811/vle_using_solvation_free_energy/development/") 
sys.path.append("/beegfs/work/st_163811/vle_using_solvation_free_energy/")

from tools.reader import get_dh_dl, get_data
from utils_automated import change_inputfile, change_jobfile, trackJobs, train_gpr, get_new_lambdas, get_partial_uncertanty, get_rmsd

# Define root 
root = "/home/st/st_st/st_st163811/workspace/vle_using_solvation_free_energy/development/lammps/butylamine_butylamine/adapted/"

## Define logger ## 

# Delete old logger file if exists
if os.path.exists(root+'opt_intermediates.log'): os.remove(root+'opt_intermediates.log')

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

# Define atom types of couple molecule 
atom_list_coulped_molecule = [1,2,3,4,5]

# Define numerical pertubation
delta          = 0.001

# Define minimum and maximum number of simulations
N_min          = 10
N_max          = 15

# Define after which rmsd convergence is achieved
tolerance      = 0.3

# Start with an initial simulation of the system
l_init = [ 0.0, 0.4, 1.0]

# Define original input file for simulation setup

os.makedirs( root+"plots", exist_ok=True )
os.makedirs( root+"run_files", exist_ok=True )

orig_file      = root+"../lammps.in"
file_output    = root+"sim_vdw_%d/lammps.in"
job_file       = root+"run.sh"
job_output     = root+"run_files/run_%d.sh"
data_output    = root+"sim_vdw_%d/fep.sampling"
idx_sim_folder = []

# Setup original simulation folder
for i,li in enumerate(l_init):
    logger.info("Creating job files:\n %s"%(file_output%i))
    change_inputfile(orig_file, file_output%i, li, atom_list_coulped_molecule, restart=False)
    idx_sim_folder.append(i)

# Submit simulations
job_list = []
for i in idx_sim_folder:
    # Write job file
    change_jobfile(job_file,job_output%i,"sim_vdw_%d"%i)
    
    # Submit job file
    exe = subprocess.run(["qsub",job_output%i],capture_output=True,text=True)
    job_list.append( exe.stdout.split("\n")[0] )

logger.info("These are the submitted jobs:\n" + " ".join(job_list) + "\nWaiting until they are finished...")

# Let python wait for the jobs to be finished (check job status every 5 min and if all jobs are done
# didnt check why they are done, then continue with this script
trackJobs(job_list)

logger.info("\nJobs are finished! Continue with postprocessing\n")

## Start iterating that at least N_min intermediates are taken and the rmsd is lower than the tolerance ##

# Define learning input for GPR 
l_learn   = np.array([[]])
var_learn = np.array([[]])

# Define starting paramters for iteration
N_sim = len(l_init)
rmsd  = 1.0
itter = 0

while N_sim <= N_max:

    N_sim = len(l_init)

    logger.info("\nIteration %d with current n° of intermediates: %d\n"%(itter,N_sim))

    # Gather simulation results
    paths     = [data_output%i for i in idx_sim_folder]
    mean, var = get_dh_dl( fe_data = [get_data(paths)], no_intermediates = N_sim, delta = delta , both_ways = False)

    logger.info("Current lambda intermediates:\n" + " ".join([str(np.round(l,3)) for l in l_init]) + "\n")
    logger.info("Current dU/dlambda simulation results:\n" + " ".join([str(np.round(m,3)) for m in mean]) + "\n")
    logger.info("Current var simulation results:\n" + " ".join([str(np.round(v,3)) for v in var]) + "\n")

    # Compute the RMSD with current simulation results
    dG_i      = get_partial_uncertanty( l_init, var )
    rmsd      = get_rmsd( dG_i )

    logger.info("Uncertanties of current deltaG_i:\n" + "\t".join(["var( delta G(%d -> %d) ): %.3f"%(i,i+1,dgi) for i,dgi in enumerate(dG_i)]) + "\n")
    logger.info("\nLeading to a current RMSD of %.1f%%\n"%(rmsd*100))

    # End the iteration if either the tolerance and the minimum number of simulations is achieved 
    if not( N_sim < N_min or rmsd > tolerance ): 
        break

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
    
    # Plot GPR model if N_sim is a multiple of 2

    if N_sim % 2 == 0:
        x_plot = np.linspace(0,1,51).reshape(-1,1)
        m, v   = gpr_modeling.predict(x_plot)
        plt.plot( gpr_modeling.X.flatten(), gpr_modeling.Y.flatten(), marker=".", linestyle="None", color="black", label="learning data (N$_\mathrm{learn}$=%d)"%len(l_learn) )
        plt.plot( x_plot.flatten(), m.flatten(), color="tab:blue", label="prediction" )
        plt.fill_between( x_plot.flatten(), m.flatten()+np.sqrt(v).flatten(), m.flatten()-np.sqrt(v).flatten(), alpha = 0.3, color="tab:blue" )
        plt.legend()
        plt.xlim(0,1)
        plt.ylim(-0.5,1.1*np.max(gpr_modeling.Y))
        plt.savefig( root+"plots/gpr_%d.png"%(itter) )
        plt.close()

        # write out training data (just for debug reasons)

        with open(root+"plots/training_data.txt" ,"a") as f: 
            f.write( "\nCurrent number of intermediates: %d"%N_sim )
            f.write( "\nCurrent training locations:\n" + " ".join([str(np.round(m,3)) for m in gpr_modeling.X.flatten()]) + "\n" )
            f.write( "\nCurrent training data:\n" + " ".join([str(np.round(m,3)) for m in gpr_modeling.Y.flatten()]) + "\n" )
            f.write("\n\n")

    # Get new lambdas (this means redistribute the current lambdas and add one point more) using a GPR and minimization method (BO or nelder-mead) or interpolation redistribution
    l_init = get_new_lambdas(lambdas = l_init, variances = var, method = "linear_adapt", precision=2, verbose=True)

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
        
        # If lambda was not simulated bevore
        else:
            os.mkdir( sim_folder_names%nli )
            

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
            change_inputfile(orig_file, file_output%idx, l, atom_list_coulped_molecule, restart=True,
                             restart_file="../sim_vdw_%d/equil.restart"%idx_nearest_lambda,
                             nearest_lambda=nearest_lambda)
            idx_sim_folder.append( idx )

            #print("\nlambda %.2f does not exist, create new one"%l)
            #print("using %.2f as nearest restart. with sim_folder %d\n"%(nearest_lambda,idx_nearest_lambda))

    # Submit simulations
    job_list = []
    for i in idx_sim_folder:
        # Write job file
        change_jobfile(job_file,job_output%i,"sim_vdw_%d"%i)

        # Submit job file
        exe = subprocess.run(["qsub",job_output%i],capture_output=True,text=True)
        job_list.append( exe.stdout.split("\n")[0] )

    logger.info("These are the submitted jobs:\n" + " ".join(job_list) + "\nWaiting until they are finished...")

    # Let python wait for the jobs to be finished (check job status every 5 min and if all jobs are done
    # didnt check why they are done, then continue with this script
    trackJobs(job_list)
    
    logger.info("\nJobs are finished! Continue with postprocessing\n")

    # merge already simulated and new simulated idx list to evaluate correctly in new iteration 
    idx_sim_folder += idx_old_folder
    idx_sim_folder.sort()

    itter +=1

logger.info("\n###############################################\n")
logger.info("\nOptimizing is finished!\n")
logger.info("\nThe optimized intermediates are:\n[ " + ", ".join([str(l) for l in l_init]) + " ]\n")
logger.info("\nThe job folders indices are:\n[ " + ", ".join([str(i) for i in idx_sim_folder]) + " ]\n")

# Gather last time the simulation results and process them
paths     = [data_output%i for i in idx_sim_folder]
mean, var = mean, var = get_dh_dl( fe_data = [get_data(paths)], no_intermediates = N_sim, delta = delta , both_ways = False)
    
logger.info("Current dU/dlambda simulation results:\n" + " ".join([str(np.round(m,3)) for m in mean]) + "\n")
logger.info("Current var simulation results:\n" + " ".join([str(np.round(v,3)) for v in var]) + "\n")

dG_i      = get_partial_uncertanty( l_init, var )

logger.info("Partial uncertanties:\n" + " ".join([str(np.round(dg,3)) for dg in dG_i]) + "\n")

rmsd      = get_rmsd( dG_i )

logger.info("Final RMSD of partial uncertanties: %.2f%%"%(rmsd*100))

## Plot last trained gp model ##
x_plot = np.linspace(0,1,51).reshape(-1,1)
m, v   = gpr_modeling.predict(x_plot)
plt.plot( gpr_modeling.X.flatten(), gpr_modeling.Y.flatten(), marker=".", linestyle="None", color="black", label="learning data (N$_\mathrm{learn}$=%d)"%len(l_learn) )
plt.plot( x_plot.flatten(), m.flatten(), color="tab:blue", label="prediction" )
plt.fill_between( x_plot.flatten(), m.flatten()+np.sqrt(v).flatten(), m.flatten()-np.sqrt(v).flatten(), alpha = 0.3, color="tab:blue" )
plt.legend()
plt.xlim(0,1)
plt.ylim(-0.5,1.1*np.max(gpr_modeling.Y))
plt.savefig( root+"plots/gpr_finished.png" )
plt.close()

logger.info("\n###############################################\n")
    
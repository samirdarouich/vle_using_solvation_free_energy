#!/usr/bin/env python3.10
#PBS -q long
#PBS -l nodes=1:ppn=1
#PBS -l walltime=96:00:00
#PBS -j oe
#PBS -N opt_intermediate_python
#PBS -o /beegfs/work/st_163811/marcelle/co2_wat/05_free_energy/05_iterative/LOG_adapt
#PBS -l mem=1000mb

import os
import sys
import logging
import subprocess
import numpy as np

sys.path.append( '/beegfs/work/st_163811/marcelle/adapt_lambdas' )

from utils import ( get_gauss_legendre_points_intermediates, get_mbar, convergence, get_unified_lambdas, 
                    change_number_of_states, get_gridpoint_function, adapted_distribution, trackJobs,
                    restart_configuration, iteration_jobfiles, new_mdp_free_energy, initial_jobfiles )


## Define logger ## 

# Define root
root     = '/beegfs/work/st_163811/marcelle'

# Define simulation folder
sim_path = f'{root}/co2_wat/05_free_energy/05_iterative'


# Delete old logger file if exists
if os.path.exists(f'{sim_path}/opt_intermediates.log'): os.remove(f'{sim_path}/opt_intermediates.log')

# Create a logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)

# Create a file handler and set the level to debug
file_handler = logging.FileHandler(f'{sim_path}/opt_intermediates.log')
file_handler.setLevel(logging.INFO)

# Create a formatter and set the format for log messages
formatter = logging.Formatter('%(asctime)s -  %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

#### Simulations settings ####

# Define template paths
mdp_template = f'{root}/adapt_lambdas/template_sim.mdp'
job_template = f'{root}/adapt_lambdas/template_decoupling.gmx'

# Define initial input for free energy simulations
initial_topo = f'{root}/co2_wat/00_topo/topol_Perkins_C00_CO2.top'
intial_coord = f'{root}/co2_wat/04_plain_MD/MD/npt_pr_298.tpr'
intial_cpt   = f'{root}/co2_wat/04_plain_MD/MD/npt_pr_298.cpt'
job_name     = "co2_wat"

# Define contstraints used in the system
constraint_settings  = { "constraints": "h-bonds", "lincs_order": 4, "lincs_iter": 2 }

# Define simulation steps in the order they are executed (prod == npt)
sim_steps    = [ "em", "nvt", "npt", "prod" ]

# Define temperature, pressure, and compressibility of the system
temperature     = 298.15
pressure        = 1.0
compressibility = 4e-5

# Define cut off in nm
rcut         = 1.5

# Integration timestep of the simulation given in ps
dt           = 0.002

# Simulation time for each simulation phase in ns
time        = { "nvt": 2, "npt": 2, "prod": 10, "iteration": 0.2 }

# Define coupled molecule type and intermediate lambdas (first decouple coulomb then vdW)
couple_moltype = "CO2"

# Vdw and Coulomb intermediates
precision  = 3

points, weights = get_gauss_legendre_points_intermediates(0.0,1.0,7)

lambdas_vdw  = np.round(points, precision)

points, weights = get_gauss_legendre_points_intermediates(0.0,1.0,4)

lambdas_coul = np.round(points, precision)

combined_lambdas = np.concatenate( (lambdas_coul,lambdas_vdw+1) )

logger.info( f"Initial vdW intermediates: { ' '.join( [ f'{max(l-1,0.0):.{precision}f}' for l in combined_lambdas] ) }\n" )
logger.info( f"Initial coul intermediates: { ' '.join( [ f'{min(l,1.0):.{precision}f}' for l in combined_lambdas] ) }\n" )

# Write mdp files 
mdp_files  = []
for step in sim_steps:

    steps = int(time[step]*1e3/dt) if step != "em" else 0

    # Create mdp files for each intermediate
    mdp_files.append( new_mdp_free_energy( mdp_template = mdp_template, destination_path = f"{sim_path}/iteration_0", sim_type = step, 
                                           new_lambdas = combined_lambdas, couple_moltype = couple_moltype, temperature = temperature,
                                           pressure = pressure, steps = steps, dt = dt, constraint_settings = constraint_settings,
                                           rcut = rcut, compressibility = compressibility ) )

# Write job files for each intermediate
initial_jobfiles( job_template = job_template, destination_path= f"{sim_path}/iteration_0", lambdas = combined_lambdas , 
                  mdp_files = mdp_files, intial_coord = intial_coord, intial_cpt = intial_cpt, initial_topo = initial_topo,
                  job_name = job_name, sim_steps = sim_steps )


# Define iteration to start and tolerance (of relative rmsd) when to stop
iteration = 1
tolerance = 0.2

if iteration == 0:
    # Submit intial simulations
    logger.info("Submit initial simulations")
    job_list = []
    for i in range( len(combined_lambdas) ):
        # Submit job file
        exe = subprocess.run(["qsub",f"{sim_path}/iteration_0/job_files/job_{i}.sh"],capture_output=True,text=True)
        job_list.append( exe.stdout.split("\n")[0] )

    logger.info("These are the submitted jobs:\n" + " ".join(job_list) + "\nWaiting until they are finished...")

    # Let python wait for the jobs to be finished (check job status every 1 min and if all jobs are done
    trackJobs(job_list)

    logger.info("\nJobs are finished! Continue with postprocessing\n")


while iteration <= 10:

    logger.info( f"Iteration n°{iteration}\n" )

    # Initialize the MBAR object
    mbar = get_mbar( path = f"{sim_path}/iteration_{iteration}", production = "prod", temperature = temperature )

    logger.info( f"Free energy difference: {mbar.delta_f_.iloc[-1,0] * 8.314 * temperature / 1000 :.3f} ± {mbar.d_delta_f_.iloc[-1,0] * 8.314 * temperature / 1000 :.3f} kJ/mol\n" )

    # Check convergence, if achieved stop.
    rmsd_rel = convergence( mbar )

    logger.info( f"Relative devation to equal partial free energy uncertainties: {rmsd_rel*100 :.0f}\n" )

    # End the loop if relative rmsd is below tolerance
    if rmsd_rel < tolerance : 
        break

    # Get the lambdas of previous iteration
    unified_lambdas     = get_unified_lambdas( mbar )

    logger.info( f"Previous vdW intermediates: { ' '.join( [ f'{max(l-1,0.0):.{precision}f}' for l in unified_lambdas] ) }\n" )
    logger.info( f"Previous coul intermediates: { ' '.join( [ f'{min(l,1.0):.{precision}f}' for l in unified_lambdas] ) }\n" )

    # Check overlapp matrix to define if a intermediate need to be added / removed or kept constant
    delta_intermediates = change_number_of_states( mbar )

    logger.info( f"The number of intermediates {'remains the same' if delta_intermediates == 0 else 'is increased by one' if delta_intermediates == 1 else 'is decreased by one' }.\n" )

    # Get gridpoint function of free energy uncertanties
    gridpoint_function  = get_gridpoint_function( mbar )

    # Define new lambda states, the new number of intermediates depends on the overlapp matrix
    combined_lambdas    = adapted_distribution( unified_lambdas, gridpoint_function, len(unified_lambdas)+delta_intermediates)

    logger.info( f"New vdW intermediates: { ' '.join( [ f'{max(l-1,0.0):.{precision}f}' for l in combined_lambdas] ) }\n" )
    logger.info( f"New coul intermediates: { ' '.join( [ f'{min(l,1.0):.{precision}f}' for l in combined_lambdas] ) }\n" )

    # Make new mdp files and simulations using the newly adapted lambdas. Simulate each 
    new_mdp_free_energy( mdp_template = mdp_template, destination_path = f"{sim_path}/iteration_{iteration+1}", sim_type = "prod", 
                         new_lambdas = combined_lambdas, couple_moltype = couple_moltype, temperature = temperature,
                         pressure = pressure, steps = int(time["iteration"]*1e3/dt), dt = dt,
                         constraint_settings = constraint_settings, rcut = rcut, compressibility = compressibility )

    # Search the indices of the intermediate states that fit the most to the new adapted lambdas
    restart_indices = restart_configuration( unified_lambdas, combined_lambdas) 

    # Create job files
    iteration_jobfiles( job_template = job_template, destination_path = sim_path, iteration = iteration+1, 
                        initial_topo = initial_topo, production = "prod", restart_indices = restart_indices )
    
    # Submit simulations
    job_list = []
    for i in range( len(combined_lambdas) ):
        # Submit job file
        exe = subprocess.run(["qsub",f"{sim_path}/iteration_{iteration+1}/job_files/job_{i}.sh"],capture_output=True,text=True)
        job_list.append( exe.stdout.split("\n")[0] )

    logger.info("These are the submitted jobs:\n" + " ".join(job_list) + "\nWaiting until they are finished...")

    # Let python wait for the jobs to be finished (check job status if all jobs are done)
    trackJobs(job_list)
    
    logger.info("\nJobs are finished! Continue with postprocessing\n")

    iteration += 1
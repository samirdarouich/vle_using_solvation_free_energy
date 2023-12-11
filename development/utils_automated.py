#!/usr/bin/env python3.10

import os
import subprocess
import GPy
import numpy as np
import matplotlib.pyplot as plt
import sys

from GPyOpt.methods import BayesianOptimization
from pymbar import timeseries

sys.path.append("/home/st/st_st/st_st163811/workspace/vle_using_solvation_free_energy")

from tools.utils import get_data, cubic_integration
from tools.gaussian_process import GPR


def get_mean_sim_data_lo(paths,b,verbose=False):
    fe_data = get_data(paths,{"adapt":False,"fraction":0.0})
    val    = []
    var    = []
    
    for inter in range( b ):

        val_exp    = fe_data[2][inter]
        val_v      = fe_data[3][inter]

        values     = -np.log( val_exp / val_v ) / 0.0001

        indices    = timeseries.subsample_correlated_data(values)
        
        std_pymbar = np.std( values[indices] ) / np.sqrt( len(indices) - 1 )
        
        if verbose: print("Number of uncorelated samples: %d. Percentage of total values: %.2f%%"%(len(indices),len(indices)/len(values)*100))
        
        val.append( np.mean( values[indices] ) )
        var.append( std_pymbar**2 )
                   
    return val, var        


def train_gpr(lambda_list,var):

    l_learn      = lambda_list if np.array(lambda_list).ndim == 2 else np.atleast_2d(lambda_list).T
    v_learn      = var if np.array(var).ndim == 2 else np.atleast_2d(var).T
    gpr_modeling = GPR( l_learn, v_learn, kernel=GPy.kern.RBF, n_optimization_restarts=3 )

    gpr_modeling.model.Gaussian_noise.variance = 0.0
    gpr_modeling.model.Gaussian_noise.variance.fix()

    gpr_modeling.train()

    return gpr_modeling


def g_lambda_i(partial_uncertainty):
    # Function that creates gridpoint function g(lambda_i)
    gridpoint_function = []	

    gridpoint_function.append(0.0)
    for i in range(len(partial_uncertainty)):
        gridpoint_function.append(round(partial_uncertainty[i] + gridpoint_function[i], 3))

    return gridpoint_function

def adapted_distribution(lambda_list, gridpoint_function, nstates): 
    # Defines new lambda-states (ranging from 0 to 1) by linear interpolation approach for equal partial uncertainties

	adapted_distribution = []
	average = 0

    # This average is the same as average of std_dG_i, excpet rounding error in gridpoint fucntion (rounded after 3 decimals)
    # Ofc this is only true if nstates is the same as the previous sampled points for std_dG_i
	average = max(gridpoint_function)/(nstates-1)
	
    # To get new distribution, interpolate the current cumultative g(lambda_i) over the gridpoints and then evaluate the new lambda
    # that is needed to match that the uncertanty is i*dG_opt
	for i in range(nstates):		
		adapted_distribution.append(float(round(np.interp(i*average, gridpoint_function, lambda_list),3)))

	return adapted_distribution

def get_partial_uncertanty(lambda_list,var_list):

    ## Compute the uncertanty of the free energy differences between each intermediate and ##
    ## Therefore integrate dummy array (zeros) to get weights and the variances of each intermediate ##

    std_dG_i = []

    for i in range(len(lambda_list)-1):
        tmp = cubic_integration( lambda_list[i:i+2], np.zeros(2), var_list[i:i+2]  )
        std_dG_i.append( tmp[1] )

    ## Return new list of std of dG_i ##

    return std_dG_i


def get_rmsd(std_dG_i,verbose=False):

    # var_dG_opt equals average of all partial uncertanties
    if verbose: print("\nOptimal uncertanty in each lambda step: %.3f\n"%np.average(std_dG_i))

    # Relative root mean square deviation
    # This should be minized that every lambda step has an equal variance
    rmsd = np.std(std_dG_i, ddof=1) / np.average(std_dG_i)
    if verbose: print("RMSD to optimal value: %.2f%%\n"%(rmsd*100))

    return rmsd

def estimate_variance( x, model, x_list, var_list, verbose=False ):

    ## Predict dU/dlambda variance of given point using a trained GPR ##

    # Incase GPR prediction is not good trained, ensure that only positive variances are predicted
    var,var_var  = model.predict( x )
    var          = np.abs(var) 
    
    if verbose: print( "Location: %.3f --> predicted var: %.3f ± %.3f \n"%( x[0][0], var[0][0], np.sqrt(var_var[0][0]) ) )
    
    x_l   = list(x_list).copy()
    var_l = list(var_list).copy()

    x_l.append( x[0][0] )
    var_l.append( var[0][0] )

    idx   = np.argsort(x_l)
    x_l   = np.array(x_l)[idx]
    var_l = np.array(var_l)[idx]

    return get_partial_uncertanty(x_l,var_l)


def trackJobs(jobs, waittime=60):
    while len(jobs) != 0:
        for jobid in jobs:
            x = subprocess.run(['qstat', jobid],capture_output=True,text=True)
            # Check wether the job is finished but is still shuting down
            try:
                dummy = " C " in x.stdout.split("\n")[-2]
            except:
                dummy = False
            # Or if it's already done (then an error fill occur)
            if dummy or x.stderr:
                jobs.remove(jobid)
                break
        os.system("sleep " + str(waittime))
    return

def change_inputfile(input_file,output,lambda_point,atom_list_coulped_molecule,restart=False,restart_file="",nearest_lambda=0.0):

    os.makedirs( os.path.dirname(output), exist_ok=True )

    with open(input_file) as f_inp: lines_inp = [line for line in f_inp]

    # Change that restart file is read in instead of data file
    if restart:
        idx = [i for i,line in enumerate(lines_inp) if "read_data" in line][0]
        new = "read_restart %s # nearest lambda: %.3f"%(restart_file,nearest_lambda)
        lines_inp[idx] = new

        # also prevent new velocity creation
        idx = [i for i,line in enumerate(lines_inp) if "velocity" in line][0]
        lines_inp[idx] = "#"+lines_inp[idx]


    # Change lambdas to specified point
    idx_lj   = [i for i,line in enumerate(lines_inp) if all([("lj/cut/soft" in line or "mie/cut/soft" in line),"pair_coeff" in line])]
    
    for idx in idx_lj:
        line = lines_inp[idx].split()
        i, j = int(line[1]), int(line[2])
        lambda_idx = 8 if "mie/cut/soft" in line else 6

        if (i in atom_list_coulped_molecule and not j in atom_list_coulped_molecule) or (not i in atom_list_coulped_molecule and j in atom_list_coulped_molecule):
            line[lambda_idx] = str(lambda_point)
            lines_inp[idx]   = " ".join( line ) + "\n"

    with open(output,"w") as f_out: f_out.writelines(lines_inp)

def change_jobfile(input_file,output,folder):

    with open(input_file) as f_inp: lines_inp = [line for line in f_inp]

    # Change working directory in job file
    idx = [i for i,line in enumerate(lines_inp) if "v_dir" in line][0]
    old = lines_inp[idx].split("=")[1]
    new = old.replace( old.split("/")[-1], folder )
    lines_inp[idx] = lines_inp[idx].replace( old, new+"\n" )

    # Change folder for LOG output
    idx = [i for i,line in enumerate(lines_inp) if "#PBS -o" in line][0]
    old = lines_inp[idx].split()[2]
    new = old.replace( old.split("/")[-2], folder )
    lines_inp[idx] = lines_inp[idx].replace( old, new )

    # Change job name
    idx = [i for i,line in enumerate(lines_inp) if "#PBS -N" in line][0]
    old = lines_inp[idx].split()[2]
    lines_inp[idx] = lines_inp[idx].replace( old, folder )

    with open(output,"w") as f_out: f_out.writelines(lines_inp)


def get_new_lambdas(lambdas,variances,gpr_model,method="BO",precision=3,logger=None,verbose=False):

    if method=="BO":

        # Set initial lambdas to 0.0 and 1.0 as they are always needed
        # use the (on all data) trained GPR model and redistibute the current lambdas to minimize the rmsd in each step 
        # this avoids that the initial simulations has to be kept, as they might not be at a relevant place

        # --> to redistribute formally each point has to be simulated and its variance has to be sampled  
        # --> this is replaced by letting the GPR approximate the simulated variance, and thus let the BO find optimial 
        # intermediates based on the GPR prediction of the real simulation variance --> take absolut value to avoid negative variances

        domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (0.01,0.99)}]

        lambda_init = [0.0, 1.0]
        var_init    = [ abs( gpr_model.predict([[xx]])[0].item() ) for xx in lambda_init ]

        if verbose:
            logger.info("\nOptimizing the intermediate points using Bayesian optimization")
            logger.info("\nCurrent intermediate points: \n" + "  ".join( [str(l) for l in lambda_init] ) )

        # redistribute current lambdas (len(lambdas)-2, as we have 2 inital lambdas fixed) and add one more +1
        for _ in range(len(lambdas)-2+1):
            
            # Define objective function with current data
            def my_objective_function(x):

                # Get standard deviations of free energy differences by approximating the simulation variance of point x with a GPR
                # and then compute free energy differences between each point --> here only the variance of the free energy difference
                # Use this to evaluate the rmsd for a flat variance distribution

                # Cut the test lambdas to only have few decimals, to avoid oversampling later on (like 0.192, 0.193)
                x        = np.round( x, precision )

                # To avoid nan due to same lambdas (e.g: 0.481 and 0.482 --> both round to 0.48, return very high rmsd)
                if x in lambda_init:
                    rmsd = 2
                else:
                    std_dG_i = estimate_variance( x, gpr_model, lambda_init, var_init,  )

                    ## Return RMSD for optimal uncertanty in each window ##

                    rmsd     = get_rmsd( std_dG_i )
                
                return rmsd
            
            # Run the optimization to search for the next intermediate 
            
            bo     = BayesianOptimization( f=my_objective_function, domain=domain, acquisition_type='EI', exact_feval=True )
            bo.run_optimization( max_iter=20 )

            # Add new lambda along with the gpr prediction of its variance
            # The mecklenfeld approach does this approximation with linear interpolation
            lambda_init.append( np.round( bo.x_opt[0], precision ) )
            lambda_init.sort()
            
            var_init = [ abs( gpr_model.predict([[xx]])[0].item() ) for xx in lambda_init ]

            if verbose:
                logger.info("\nThe minumum RMSD was %.3f (x = %.3f)" % (bo.fx_opt, bo.x_opt))
                logger.info("\nCurrent intermediate points: \n" + "  ".join( [str(l) for l in lambda_init] ) )
                logger.info("GPR estimated variance of current points: \n" + "  ".join( [str(np.round(l,5)) for l in var_init] ) )

    else:
        # Use meckenfeld scheme to adapt lambdas to match linear interpolation scheme
        
        if verbose:
            logger.info("\nOptimizing the intermediate points using linear interpolation")
        # get partial uncertanties
        dG_i_inter  = get_partial_uncertanty(lambdas,variances)
        
        # Get rmsd to optimal variance
        rmsd        = get_rmsd(dG_i_inter)

        # get the grid function
        grid        = g_lambda_i(dG_i_inter)

        # get new distribution with one lambda more as before
        lambda_init = adapted_distribution(lambdas,grid,len(lambdas)+1)

        if verbose:
            logger.info("\nThe current RMSD is %.3f (x = %.3f)"%(rmsd*100))
            logger.info("\nCurrent intermediate points: \n" + "  ".join( [str(l) for l in lambda_init] ) )


    return lambda_init
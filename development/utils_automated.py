#!/usr/bin/env python3.10

import os
import GPy
import sys
import math
import logging
import subprocess
import numpy as np
from typing import List
from scipy.optimize import minimize, Bounds
from GPyOpt.methods import BayesianOptimization

sys.path.append("/home/st/st_st/st_st163811/workspace/vle_using_solvation_free_energy")

from tools.numeric_utils import naturalcubicspline
from tools.gaussian_process import GPR

logger = logging.getLogger(__name__)

def train_gpr(lambda_list,var,kernel= GPy.kern.RBF):

    l_learn      = lambda_list if np.array(lambda_list).ndim == 2 else np.atleast_2d(lambda_list).T
    v_learn      = var if np.array(var).ndim == 2 else np.atleast_2d(var).T
    gpr_modeling = GPR( l_learn, v_learn, kernel=kernel, n_optimization_restarts=3 )

    gpr_modeling.model.Gaussian_noise.variance = 0.0
    gpr_modeling.model.Gaussian_noise.variance.fix()

    gpr_modeling.train()

    return gpr_modeling


def g_lambda_i(partial_uncertainty):
    # Function that creates gridpoint function g(lambda_i)
    gridpoint_function = []	

    gridpoint_function.append(0.0)
    for i in range(len(partial_uncertainty)):
        gridpoint_function.append( round(partial_uncertainty[i] + gridpoint_function[i], 3) )

    return gridpoint_function

def adapted_distribution(lambda_list, gridpoint_function, nstates, precision: int=3): 
    # Defines new lambda-states (ranging from 0 to 1) by linear interpolation approach for equal partial uncertainties

	adapted_distribution = []
	average = 0

    # This average is the same as average of std_dG_i, excpet rounding error in gridpoint fucntion (rounded after 3 decimals)
    # Ofc this is only true if nstates is the same as the sampled points for std_dG_i
	average = max(gridpoint_function)/(nstates-1)
	
    # To get new distribution, interpolate the current cumultative g(lambda_i) over the gridpoints and then evaluate the new lambda
    # that is needed to match that the uncertanty is i*dG_opt
	for i in range(nstates):		
		adapted_distribution.append( float( round( np.interp( i*average, gridpoint_function, lambda_list), precision ) ) )

	return adapted_distribution

def get_partial_uncertanty(lambda_list,var_list):

    # Compute the uncertanty of the free energy differences between each intermediate ##
    # Therefore build the natural cubic splines for the given lambdas and integrate the squared weights with the variances.
    # The spline weights are a matrix, containing in each row i, the integration weights from lambda i to lambda i+1. 
    # The final integral is the summation over each subintegral 

    std_dG_i = []
    spl      = naturalcubicspline( np.array( lambda_list ) )

    for i in range( len(lambda_list) - 1 ):
        std_dG_i.append( np.sqrt( np.dot( spl.wk[i,:]**2, var_list ) ) )

    ## Return new list of std of dG_i ##

    return std_dG_i


def get_rmsd(std_dG_i,verbose=False):

    # std_dG_opt equals average of all partial uncertanties
    std_dG_opt = np.mean(std_dG_i)
    if verbose: print("\nOptimal uncertanty in each lambda step: %.3f\n"%std_dG_opt)

    # Reltaive root mean square deviation to the optimal value
    rmsd = np.sqrt( np.mean( (np.array(std_dG_i) - std_dG_opt)**2 ) ) / std_dG_opt

    if verbose: print("Relative RMSD to optimal value: %.2f%%\n"%(rmsd*100))

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


def trackJobs(jobs, waittime=10):
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


def new_lambdas_using_GPR(lambdas: List[float], variances: List[float], N_intermediates: int, gpr_model: GPR, optimization_method: str='nelder_mead', 
                          precision: int=3, verbose: bool=False):
    """
    This function uses a on variances of the derivative of the Hamiltonian with respect to the coupling parameter lambda trained Gaussian process regression model to 
    predict newly distributed lambda intermediates that aim to flatten the variance of the free energy difference between each lambda.

    Args:
        lambdas (List[float]): Intermediate lambdas used to compute a free energy difference
        variances (List[float]): Simulation variances for each intermediate lambda.
        N_intermediates (int): Number of redistributed intermediates.
        gpr_model (GPR): Gaussian process regression object that is trained on variances of the derivative of the Hamiltonian with respect to the coupling parameter lambda.
        optimization_method (str, optional): Methods to redistribute the intermediates. Either use a gradient free numerical 
        method such as Nelder_mead or bayesian optimization. Defaults to 'nelder_mead'.
        precision (int, optional): Number of decimals per intermediate. Defaults to 3.
        verbose (bool, optional): If detailed throughout the itreations output should be printed to a logger.
    """
    
    # Get the current partial uncertanties
    std_dG_i    = get_partial_uncertanty( lambdas, variances )

    # The aim is to flatten the variance over the whole lambda range. The optimal equalised uncertanty between each intermediate point
    # is approximated to be the average of the current uncertanties --> The average is computed in dependence of the number of newly redistributed intermediates.
    # Hence the optimal free energy standard deviation between each intermediate is the sum of all standard deviations divided by the number of newly states-1.
    # The minus one is necessary, as from 5 intermediate states, one get 4 free energy differences (from 1 to 2, 2 to 3, 3 to 4 and 4 to 5)
    std_dG_opt = np.sum( std_dG_i ) / ( N_intermediates - 1 )
    print("These are the current std (bevore iteration):",std_dG_i)
    print("This is the optimal std",std_dG_opt,"\n")
    # Use the GPR model to estimate the variance at the boundary points
    lambda_init = [0.0, 1.0]
    var_init    = [ abs( gpr_model.predict([[xx]])[0].item() ) for xx in lambda_init ]

    for _ in range(N_intermediates-2):
        
        # Define objective function with current data
        def objective_function(x):

            # Get standard deviations of free energy differences by approximating the simulation variance of point x with a GPR
            # and then compute variances of the free energy differences between each point.

            # Cut the test lambdas to only have few decimals, to avoid oversampling later on (like 0.192, 0.193)
            if np.array(x).ndim < 2:
                x = np.round( np.array(x).reshape(-1,1), precision )
            else:
                x = np.round( x, precision )

            # To avoid nan due to same lambdas (e.g: 0.481 and 0.482 --> both round to 0.48, return very high rmsd)
            if x in lambda_init:
                rmsd = 1e6
            else:
                std_dG_i = estimate_variance( x, gpr_model, lambda_init, var_init  )

                # Relative root mean square deviation of the current iteration lambdas to the optimal value 
                # (which is computed as the mean of the input data and the number of newly intermediates)
                rmsd     = np.sqrt( np.mean( (np.array(std_dG_i) - std_dG_opt)**2 ) ) / std_dG_opt

            return rmsd
            
        if optimization_method == 'nelder_mead':
            
            # Initial guess
            initial_guess = np.linspace(0.1,0.9,11)

            results = []

            for init_guess in initial_guess:
                # Perform the optimization using the Nelder-Mead method
                result = minimize( fun=objective_function, x0=init_guess, method='Nelder-Mead', bounds=Bounds(0.01,0.99)) 
                results.append(result)
            
            # The optimized intermediate of all different runs
            min_result = min(results, key=lambda result: result.fun)

            optimized_intermediate = np.round( min_result.x[0], precision )
            rmsd                   = min_result.fun * 100
        
        elif optimization_method == 'bayesian_optimization':
            domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (0.01,0.99)}]

            bo     = BayesianOptimization( f=objective_function, domain=domain, acquisition_type='EI', exact_feval=True )
            bo.run_optimization( max_iter=20 )

            # Add new lambda along with the gpr prediction of its variance
            optimized_intermediate = np.round( bo.x_opt[0], precision )
            rmsd                   = bo.fx_opt * 100

        else:
            raise KeyError(f"Specified optimization method is not implmented: {optimization_method}")
        
        # Add new lambda along with the gpr prediction of its variance
        # The mecklenfeld approach does this approximation with linear interpolation
        lambda_init.append( optimized_intermediate )
        lambda_init.sort()
        
        var_init = [ abs( gpr_model.predict([[xx]])[0].item() ) for xx in lambda_init ]

        if verbose:
            logger.info("\nThe minumum RMSD was at %.3f: %.0f %% " %(optimized_intermediate,rmsd))
            logger.info("\nCurrent intermediate points: \n" + "  ".join( [str(l) for l in lambda_init] ) )
            logger.info("GPR estimated variance of current points: \n" + "  ".join( [str(np.round(l,5)) for l in var_init] ) )

    return lambda_init

def new_lambdas_using_adapt(lambdas: List[float], variances: List[float], N_intermediates: int, precision: int=3, verbose: bool=False):
    """This function linear intepolates the standard deviations of the free energy differences between each lambda point and redistribute the lambda points to get a flat distribution.

    Args:
        lambdas (List[float]): Intermediate lambdas used to compute a free energy difference
        variances (List[float]): Simulation variances for each intermediate lambda.
        N_intermediates (int): Number of redistributed intermediates.
        precision (int, optional): Number of decimals per intermediate. Defaults to 3.
        verbose (bool, optional): If detailed throughout the itreations output should be printed to a logger.
    """
    
    if verbose:
            logger.info("\nOptimizing the intermediate points using linear interpolation")
    
    # get partial uncertanties
    dG_i_inter  = get_partial_uncertanty(lambdas,variances)
    
    # Get rmsd to optimal variance
    rmsd        = get_rmsd(dG_i_inter)

    # get the grid function
    grid        = g_lambda_i(dG_i_inter)

    # get new distribution with one lambda more as before
    lambda_init = adapted_distribution( lambdas, grid, N_intermediates, precision = precision )

    if verbose:
        logger.info("\nThe current RMSD is %.3f (x = %.3f)"%(rmsd*100))
        logger.info("\nCurrent intermediate points: \n" + "  ".join( [str(l) for l in lambda_init] ) )

    return lambda_init

def get_new_lambdas(lambdas: List[float], variances: List[float], gpr_model: GPR=None, method: str="GPR",
                    optimization_method: str='nelder_mead', precision: int=3, verbose=False):
    """
    This function uses sampled variances and lambdas to predict newly distributed lambda intermediates that aim to flatten 
    the variance of the free energy difference between each lambda. It either uses a trained Gaussian process regression or 
    a linear interpolation approach. 

    Args:
        lambdas (List[float]): Intermediate lambdas used to compute a free energy difference
        variances (List[float]): Simulation variances for each intermediate lambda.
        gpr_model (GPR): Gaussian process regression object that is trained with lambdas and there corresponding variance. Defaults to None.
        method (str, optional): Method to get new lambda intermediate. Either using the "GPR" or the "linear_adapt". Defaults to "GPR".
        optimization_method (str, optional): If GPR is choosen, which optimization method should be utilized to redistribute the new lambdas
                                             Possiblited are "nelder_mead" or "bayesian_optimization". Defaults to 'nelder_mead'.
        precision (int, optional): Number of decimals per intermediate. Defaults to 3.
        verbose (bool, optional): If detailed throughout the itreations output should be printed to a logger.

    Raises:
        KeyError: If the specified method is not implemented.

    Returns:
        lambda_init (List[float]): Newly distributed intermediates containing one intermediate more then the input list
    """

    if method == "GPR":

        lambda_init = new_lambdas_using_GPR( lambdas = lambdas, variances = variances, N_intermediates = len(lambdas)+1,  
                                             gpr_model = gpr_model, optimization_method = optimization_method, precision = precision,
                                             verbose = verbose )

    elif method == "linear_adapt":
        # Use meckenfeld scheme to adapt lambdas to match linear interpolation scheme
        
        lambda_init = new_lambdas_using_adapt( lambdas = lambdas, variances = variances, N_intermediates = len(lambdas)+1, 
                                               precision = precision )

    else:
        raise KeyError(f"Specified method is not implemented: {method}")

    return lambda_init

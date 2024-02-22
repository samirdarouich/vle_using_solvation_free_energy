import re
import os
import glob
import logging
import subprocess
import numpy as np
import pandas as pd

from typing import List, Tuple, Dict
from jinja2 import Template
from scipy.special import roots_legendre
from alchemlyb.estimators import TI, MBAR
from alchemlyb.parsing.gmx import extract_dHdl, extract_u_nk

logger = logging.getLogger("my_logger")

def get_gauss_legendre_points_intermediates(a: float, b: float, no_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Gauss-Legendre points and weights for numerical integration.

    Parameters:
        a (float): The lower bound of the integration interval.
        b (float): The upper bound of the integration interval.
        no_points (int): The number of points to generate.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the scaled points and weights.
    """
    # get gauss-legendre weights and points
    points, weights = roots_legendre(no_points)

    # Scale points and weights to the interval [a, b]
    points_scaled = 0.5 * (b - a) * points + 0.5 * (a + b)  

    # Scale weights to account for the interval change
    weights_scaled = 0.5 * (b - a) * weights

    return points_scaled, weights_scaled

    from typing import List

def get_mbar( path: str, production: str, temperature: float, pattern: str=r'lambda_(\d+)' ) -> MBAR:
    """
    Calculate the MBAR (Multistate Bennett Acceptance Ratio) estimator for a given set of free energy output files.

    Parameters:
        path (str): The path to the simulation folder.
        production (str): The name pattern of the production files.
        temperature (float): The temperature at which the simulation was performed.
        pattern (str, optional): The regular expression pattern used to extract the intermediate number from the file names. Defaults to r'lambda_(\d+)'.

    Returns:
        MBAR: The MBAR estimator object.

    Example:
        mbar = get_mbar(path='/path/to/simulation', production='prod', temperature=300.0, pattern=r'lambda_(\d+)')
    """
    # Search for all free energy output files within the simulation folder
    filelist = glob.glob(f"{path}/**/{production}*.xvg", recursive=True)

    # Sort the paths ascendingly after the intermediate number
    filelist.sort( key=lambda path: int(re.search(pattern, path).group(1)) )

    # Extract the energy differences
    u_nk = pd.concat( [ extract_u_nk(xvg, T=temperature) for xvg in filelist ] )

    # Call the MBAR class
    mbar = MBAR().fit(u_nk)

    return mbar

def get_unified_lambdas( mbar: MBAR ) -> List[float]:
    """
    Return a list of unified lambdas.

    Parameters:
        mbar (MBAR): The MBAR object containing the states.

    Returns:
        List[float]: A list of unified lambdas, where lambdas coul range from 0 to 1 and lambdas vdw range from 1 to 2.
    """
    # lambdas coul between 0 and 1, lambdas vdw between 1 and 2
    lambdas_unified = [ sum(l) for l in mbar.states_ ]
	
    return lambdas_unified

def get_gridpoint_function( mbar: MBAR, precision: int=3 ) -> List[float]: 
    """
    Calculate the gridpoint function for a given MBAR estimator.

    Parameters:
        mbar (MBAR): The MBAR estimator object.
        precision (int, optional): The number of decimal places to round the gridpoint function values to. Default is 3.

    Returns:
        List[float]: The gridpoint function values.

    """
    gridpoint_function = []	

    gridpoint_function.append(0.0)

    for i in range( len(mbar.d_delta_f_) - 1 ):
        gridpoint_function.append( round( mbar.d_delta_f_.iloc[i,i+1] + gridpoint_function[i], precision ) )
    
    return gridpoint_function    


def change_number_of_states( mbar: MBAR ) -> int: 
    """
    Determines the change in the number of lambda-states by analyzing the overlap matrix.

    Parameters:
        mbar (MBAR): The MBAR object containing the overlap matrix.

    Returns:
        int: The change in the number of lambda-states. Possible values are -1, 0, or 1.

    """

    # Determines the change in the number of lambda-states by analysing the overlap matrix
    xi = 0 
    max_overlap = 0.20
    min_overlap = 0.10
    overlap_between_neighbours = []

    for i,row in enumerate(mbar.overlap_matrix[:-1]):
        if i == 0:
            overlap_between_neighbours.append( round(row[i+1],2) )
        else:
            overlap_between_neighbours.append( round(row[i-1],2) )
            overlap_between_neighbours.append( round(row[i+1],2) )

    if min(overlap_between_neighbours) < min_overlap:
        xi = 1
    elif max(overlap_between_neighbours) > max_overlap:
        xi = -1
    else:
        xi = 0

    return int(xi)

def adapted_distribution(lambdas: List[float], gridpoint_function: List[float], nstates: int) -> List[float]:
    """
    Calculates the adapted distribution of lambda-states based on a linear interpolation approach for equal partial uncertainties.

    Parameters:
    - lambdas (List[float]): A list of lambda values.
    - gridpoint_function (List[float]): A list of gridpoint function values.
    - nstates (int): The number of states.

    Returns:
    - adapted_distribution (List[float]): A list of adapted distribution values.

    Example:
    lambdas = [0.1, 0.5, 0.9]
    gridpoint_function = [1.0, 2.0, 3.0, 4.0, 5.0]
    nstates = 4

    adapted_distribution(lambdas, gridpoint_function, nstates)
    # Output: [0.1, 0.3, 0.7, 0.9]
    """

    # Defines new lambda-states (ranging from 0 to 2) by linear interpolation approach for equal partial uncertainties
    adapted_distribution = []
    average              = max(gridpoint_function)/(nstates-1)

    for i in range(nstates):		
        adapted_distribution.append( float( round( np.interp(i*average, gridpoint_function, lambdas),3 ) ) )

    return adapted_distribution

def restart_configuration(unified_oldlambdas: List[float], unified_newlambdas: List[float]) -> List[int]:
    """
    Returns the index for the best restart configuration for the new set of lambda-states.

    Parameters:
    - unified_oldlambdas (List[float]): A list of old lambda-states.
    - unified_newlambdas (List[float]): A list of new lambda-states.

    Returns:
    - List[int]: A list of indices representing the best restart configuration for the new set of lambda-states.

    """
     
    # Returns the index for the best restart configuration for the new set of lambda-states
    restart_configur = []  

    for i in range(len(unified_newlambdas)):
        restart_configur.append( int( round( np.interp(unified_newlambdas[i], unified_oldlambdas, range(len(unified_oldlambdas)) ), 0 ) ) )

    return restart_configur 


def convergence( mbar: MBAR ) -> float:
    """
    Returns the relative RMSD needed to determine convergence of the iteration procedure.

    Parameters:
    mbar (MBAR): The MBAR object containing the d_delta_f_ attribute.

    Returns:
    float: The relative RMSD value.

    """

    # Returns the relative RMSD needed to determine convergence of the iteration procedure
    partial_uncertainty = []
    txt                 = "Partial uncertanties:\n"

    for i in range( len(mbar.d_delta_f_) - 1 ):
        partial_uncertainty.append( mbar.d_delta_f_.iloc[i,i+1] )
        txt += f"{i+1} -> {i+2}: {mbar.d_delta_f_.iloc[i,i+1]:.4f}\t"
    
    logger.info(txt+"\n")

    RMSD_rel = np.std(partial_uncertainty, ddof=1) / np.average(partial_uncertainty)

    return RMSD_rel


def fill_mdp_template( mdp_destination: str, mdp_template: str, time_settings: Dict[str, str|float]=None,  
                       nonbonded_settings: Dict[str, str|float]=None, constraint_settings: Dict[str, str|float]=None,
                       ensemble_settings: Dict[str, str|float|bool|Dict[str, str|float]]=None,
                       free_energy_settings: Dict[str, str|float]=None, restart="no" ):
    """
    Fill the given MDP template with the provided settings and save it to the specified destination.

    Parameters:
        mdp_destination (str): The path to save the filled MDP template.
        mdp_template (str): The path to the MDP template file.
        time_settings (Dict[str, str|float], optional): The time settings for the simulation. Defaults to None.
        nonbonded_settings (Dict[str, str|float], optional): The nonbonded settings for the simulation. Defaults to None.
        constraint_settings (Dict[str, str|float], optional): The constraint settings for the simulation. Defaults to None.
        ensemble_settings (Dict[str, str|float|bool|Dict[str, str|float]], optional): The ensemble settings for the simulation. Defaults to None.
        free_energy_settings (Dict[str, str|float], optional): The free energy settings for the simulation. Defaults to None.
        restart (str, optional): The restart option for the simulation. Defaults to "no".

    Returns:
        None

    Raises:
        FileNotFoundError: If the MDP template file does not exist.

    Example:
        fill_mdp_template("output.mdp", "template.mdp", time_settings={"dt": 0.001, "steps": 500000})
    """
    if time_settings == None:
        time_settings = { "dt": 0.002, "comm": 1000, "steps": 1000000, "trajectory": 0, "log": 2500 }

    if nonbonded_settings == None:
        nonbonded_settings = { "cutoff_neighbor": 1.4, "coulomb": "PME", "cutoff_coulomb": 1.4, "cutoff_vdw": 1.4 }

    if constraint_settings == None:
        constraint_settings  = { "constraints": "no", "lincs_order": 0, "lincs_iter": 0 }

    if ensemble_settings == None:
        ensemble_settings = { "t": { "tcoupl": "nose-hoover", "tau_t": 1.0, "ref_t": 298.15 }, 
                              "p": { "pcoupl": "Parrinello-Rahman", "pcoupltype": "isotropic", "tau_p": 2.0, "ref_p": 1.0, "compressibility": 4.5e-5 }
                            }
    
    simulation_settings = { "time": time_settings, "nb": nonbonded_settings, "ensemble": ensemble_settings, "constraints": constraint_settings,
                            "free_energy": free_energy_settings, "restart": restart, "seed": np.random.randint(0,1e5) }
    
    with open( mdp_template ) as f:
        template = Template( f.read() )

    rendered = template.render( ** simulation_settings ) 

    with open( mdp_destination, "w" ) as f:
        f.write( rendered )

def write_mdp_file( mdp_destination: str, mdp_template: str, ensemble: Dict[str, str|float]=None, 
                    free_energy_settings: Dict[str, str|float]=None, constraint_settings: Dict[str, str|float]=None,
                    time_settings: Dict[str, str|float]=None, rcut: float=1.5, restart: str="no" ):
    """
    The write_mdp_file function is responsible for generating an MDP file based on a provided template and various settings. 
    It allows customization of the ensemble, free energy, constraint, and time settings.

    Parameters:
        mdp_destination (str): The path to save the filled MDP template.
        mdp_template (str): The path to the MDP template file.
        ensemble (Dict[str, str|float], optional): The ensemble settings for the simulation. Defaults to None.
        free_energy_settings (Dict[str, str|float], optional): The free energy settings for the simulation. Defaults to None.
        constraint_settings (Dict[str, str|float], optional): The constraint settings for the simulation. Defaults to None.
        time_settings (Dict[str, str|float], optional): The time settings for the simulation. Defaults to None.
        rcut (float, optional): The cutoff radius for nonbonded interactions. Defaults to 1.5.
        restart (str, optional): The restart option for the simulation. Defaults to "no".

    Raises:
        KeyError: If an invalid ensemble is specified.

    Returns:
        None

    Example:
        write_mdp_file("output.mdp", "template.mdp", ensemble={"ensemble": "npt", "ref_t": 298.15, "ref_p": 1.0}, 
                    free_energy_settings={"calc_lambda_neighbors": -1}, constraint_settings={"constraints": "no"}, 
                    time_settings={"dt": 0.002})
    """

    if ensemble == None:
        ensemble = { "ensemble": "npt", "ref_t": 298.15, "ref_p": 1.0, "tcoupl": "nose-hoover", "pcoupl": "Parrinello-Rahman", "compressibility": 4.5e-5 }

    if ensemble["ensemble"] == "npt":
        ensemble_settings = { "t": { "tcoupl": ensemble["tcoupl"], "tau_t": 0.5, "ref_t": ensemble["ref_t"] }, 
                              "p": { "pcoupl": ensemble["pcoupl"], "pcoupltype": "isotropic", "tau_p": 2.0, "ref_p": ensemble["ref_p"], "compressibility": ensemble["compressibility"]}
                            }
        
    elif ensemble["ensemble"] == "nvt":
        ensemble_settings = { "t": { "tcoupl": ensemble["tcoupl"], "tau_t": 0.5, "ref_t": ensemble["ref_t"] } }

    elif ensemble["ensemble"] == "em": 
        ensemble_settings = { "em": True }

    else:
        raise KeyError(f"Wrong ensemple specified: {ensemble['ensemble']}. Valid options are: 'npt', 'nvt', 'em'")
    
    nonbonded_settings = { "cutoff_neighbor": rcut, "coulomb": "PME", "cutoff_coulomb": rcut, "cutoff_vdw": rcut }

    fill_mdp_template( mdp_destination = mdp_destination, mdp_template = mdp_template, ensemble_settings = ensemble_settings, 
                       constraint_settings = constraint_settings, free_energy_settings = free_energy_settings,
                       time_settings = time_settings, nonbonded_settings = nonbonded_settings, restart = restart )

def new_mdp_free_energy( mdp_template: str, destination_path: str, couple_moltype:str, new_lambdas: List[float], sim_type: str,
                         temperature: float=None, pressure: float=None, compressibility: float=None, steps: float=1000000, 
                         dt: float=0.002, precision: int=3, constraint_settings: Dict[str, str|float]=None, rcut: float=1.5 ) -> List[str]:
    """
    Generate new MDP files for free energy calculations.

    Parameters:
        mdp_template (str): The path to the MDP template file.
        destination_path (str): The path to save the generated MDP files.
        couple_moltype (str): The molecule type to couple during the free energy calculation.
        new_lambdas (List[float]): The list of lambda values for the free energy calculation.
        sim_type (str): The type of simulation to perform.
        temperature (float, optional): The temperature for the simulation. Defaults to None.
        pressure (float, optional): The pressure for the simulation. Defaults to None.
        compressibility (float, optional): The compressibility for the simulation. Defaults to None.
        steps (float, optional): The number of simulation steps. Defaults to 1000000.
        dt (float, optional): The time step size. Defaults to 0.002.
        precision (int, optional): The precision for lambda values. Defaults to 3.
        constraint_settings (Dict[str, str|float], optional): The constraint settings for the simulation. Defaults to None.
        rcut (float, optional): The cutoff radius for nonbonded interactions. Defaults to 1.5.

    Returns:
        List[str]: The list of paths to the generated MDP files.

    Example:
        mdp_files = new_mdp_free_energy("template.mdp", "output", "moltype", [0.0, 0.5, 1.0], "prod", temperature=298.15, pressure=1.0, compressibility=4.5e-5, steps=1000000, dt=0.002, precision=3, constraint_settings={"constraints": "no"}, rcut=1.5)
    """

    # Define free energy settings
    free_energy_settings = { "calc_lambda_neighbors": -1, "couple_lambda0": "vdw-q", "couple_lambda1": "none", "nstdhdl": 100, 
                             "sc_alpha": 0.5, "sc_power": 1, "sc_r_power": 6, "couple_moltype": couple_moltype,
                            }
    
    # List with all mdp files
    mdp_files = []
    
    # As we are decoupling first decouple coulomb and then vdW. The adapated distribution has vdW values between 1 and 2 and coulomb from 0 to 1
    free_energy_settings["init_lambda_states"] = "".join([f"{x:.0f}" + " "*(precision+2) if x < 10 else f"{x:.0f}" + " "*(precision+1) for x,_ in enumerate(new_lambdas)])
    free_energy_settings["vdw_lambdas"]        = " ".join( [ f"{max(l-1,0.0):.{precision}f}" for l in new_lambdas] )
    free_energy_settings["coul_lambdas"]       = " ".join( [ f"{min(l,1.0):.{precision}f}" for l in new_lambdas] )

    # Define ensemble settings depending on the simulation type
    ensemble = { "ensemble": "npt" if sim_type == "prod" else sim_type,  
                 "pcoupl": "Parrinello-Rahman",
                 "tcoupl": "nose-hoover", "ref_t": temperature, "ref_p": pressure,
                 "compressibility": compressibility  }

    time_settings = { "dt": dt, "comm": 1000, "steps": steps, "trajectory": 0, "log": 10000 }

    # Define if new velocities are generated or taken by restart
    restart       = "yes" if sim_type == "npt" or sim_type == "prod" else "no"

    # Write new mdp files
    os.makedirs( f"{destination_path}/mdp_files/{sim_type}", exist_ok = True )

    for j in range(len(new_lambdas)):
        free_energy_settings["init_lambda_state"] = j
  
        mdp_destination = f"{destination_path}/mdp_files/{sim_type}/{sim_type}_{j}.mdp"
        
        write_mdp_file( mdp_destination = mdp_destination, mdp_template = mdp_template, ensemble = ensemble,
                        time_settings = time_settings, constraint_settings = constraint_settings,
                        free_energy_settings = free_energy_settings, rcut = rcut, restart = restart )

        mdp_files.append( mdp_destination )

    return mdp_files

def initial_jobfiles( job_template: str, destination_path: str, lambdas: List[float], mdp_files: List[List[str]], intial_coord: str, intial_cpt: str,
                      initial_topo: str, job_name: str, sim_steps: List[str]=[ "em", "nvt", "npt", "prod" ], pattern: str='lambda' ):
    """
    Generate initial job files for a set of simulations with different lambda values.

    Parameters:
        job_template (str): Path to the job template file.
        destination_path (str): Path to the destination folder where the job files will be created.
        lambdas (List[float]): List of lambda values.
        mdp_files (List[List[str]]): List of lists containing the paths to the MDP files for each simulation phase.
        intial_coord (str): Path to the initial coordinate file.
        intial_cpt (str): Path to the initial checkpoint file.
        initial_topo (str): Path to the initial topology file.
        job_name (str): Name of the job.
        sim_steps (List[str], optional): List of simulation steps. Defaults to ["em", "nvt", "npt", "prod"].
        pattern (str, optional): Pattern for the simulation folder names. Defaults to 'lambda'.

    Returns:
        None

    Raises:
        FileNotFoundError: If the job template file does not exist.
        FileNotFoundError: If any of the MDP files does not exist.
        FileNotFoundError: If the initial coordinate file does not exist.
        FileNotFoundError: If the initial checkpoint file does not exist.
        FileNotFoundError: If the initial topology file does not exist.
    """

    # Check if job template file exists
    if not os.path.isfile( os.path.abspath(job_template) ):
        raise FileNotFoundError(f"Job template file { os.path.abspath( job_template ) } not found.")

    # Check if topology file exists
    if not os.path.isfile( os.path.abspath(initial_topo) ):
        raise FileNotFoundError(f"Topology file { os.path.abspath( initial_topo ) } not found.")

    with open(job_template) as f:
        template = Template(f.read())

    job_file_settings = { step:{} for step in sim_steps }

    # i is the indice of the new lambda steps, j is the indice of the nearest lambda from previous iteration for each new lambda i.
    for i,_ in enumerate( lambdas ):

        # New simulation folder
        new_folder      = f"{destination_path}/{pattern}_{i}"

        # Create the new folder
        os.makedirs( new_folder, exist_ok = True )

        # Relative paths for each mdp file for each simulation phase
        mdp_relative  = [ os.path.relpath( mdp_files[j][i], f"{new_folder}/{step}" ) for j,step in enumerate(sim_steps) ]

        # Relative paths for each coordinate file (for energy minimization use initial coodinates, otherwise use the preceeding output)
        cord_relative = [ f"../{sim_steps[j-1]}/{sim_steps[j-1]}{i}.tpr" if j > 0 else os.path.relpath( intial_coord, f"{new_folder}/{step}" ) for j,step in enumerate(sim_steps) ]

        # Relative paths for each checkpoint file 
        cpt_relative  = [ f"../{sim_steps[j-1]}/{sim_steps[j-1]}{i}.cpt" if j > 0 else os.path.relpath( intial_cpt, f"{new_folder}/{step}" ) for j,step in enumerate(sim_steps) ]

        # Relative paths for topology
        topo_relative = [ os.path.relpath( initial_topo, f"{new_folder}/{step}" ) for j,step in enumerate(sim_steps) ]

        # output file for the current lambda i
        out_relative  = [ f"{step}{i}.tpr -maxwarn 10" for step in sim_steps]

        for j,step in enumerate(sim_steps):
            job_file_settings[step]["name"]   = step

            # If preceeding step is energy minimization, there is no cpt file to read in
            if sim_steps[j-1] == "em":
                job_file_settings[step]["grompp"] = f"-f {mdp_relative[j]} -c {cord_relative[j]} -p {topo_relative[j]} -o {out_relative[j]}"
            else:
                job_file_settings[step]["grompp"] = f"-f {mdp_relative[j]} -c {cord_relative[j]} -p {topo_relative[j]} -t {cpt_relative[j]} -o {out_relative[j]}"
            job_file_settings[step]["mdrun"]  = f"-deffnm {step}{i}" 

        # Define job name and output
        job_name_i = f"{job_name}_{i}"
        log_path   = f"{new_folder}/LOG"

        # Rander template
        settings = { **job_file_settings, "lambda_init": i, "job_name": job_name_i, "log_path": log_path, "working_path": new_folder }
        
        rendered = template.render( ** settings )

        # Create the job folder
        os.makedirs( f"{destination_path}/job_files/", exist_ok = True )

        # Write new job file
        with open( f"{destination_path}/job_files/job_{i}.sh", "w") as f:
            f.write( rendered )


def iteration_jobfiles( job_template: str, destination_path: str, iteration: int, production: str, restart_indices: List[int], initial_topo: str, pattern: str='lambda' ):
    """
    Generate job files for each iteration of a simulation.

    Parameters:
        job_template (str): Path to the job template file.
        destination_path (str): Path to the destination folder where the job files will be created.
        iteration (int): The current iteration number.
        production (str): The name of the production file.
        restart_indices (List[int]): List of indices indicating the nearest lambda from the previous iteration for each new lambda.
        initial_topo (str): Path to the topology file.
        pattern (str, optional): The pattern to be used in the folder and file names (default is 'lambda').

    Returns:
        None

    Raises:
        FileNotFoundError: If the job template file is not found.
        FileNotFoundError: If the topology file is not found.

    Example:
        iteration_jobfiles('job_template.txt', '/path/to/destination', 2, 'prod', [0, 1, 2], 'topology.pdb', 'lambda')
    """

    # Check if job template file exists
    if not os.path.isfile( os.path.abspath(job_template) ):
        raise FileNotFoundError(f"Job template file { os.path.abspath( job_template ) } not found.")

    # Check if topology file exists
    if not os.path.isfile( os.path.abspath(initial_topo) ):
        raise FileNotFoundError(f"Topology file { os.path.abspath( initial_topo ) } not found.")
    
    with open(job_template) as f:
        template = Template(f.read())

    # i is the indice of the new lambda steps, j is the indice of the nearest lambda from previous iteration for each new lambda i.
    for i,j in enumerate( restart_indices ):

        # New simulation folder
        new_folder      = f"{destination_path}/iteration_{iteration}/{pattern}_{i}"

        # Simulation folder from previous iteration that matches the current lambda the best
        matching_folder = f"{destination_path}/iteration_{iteration-1}/{pattern}_{j}"

        # mdp file pointing on the current iteration mdp files (relative path)
        mpd_file  = os.path.relpath( f"{destination_path}/iteration_{iteration}/mdp_files/{production}/prod_{i}.mdp", f"{new_folder}/{production}" )

        # coordinate file pointing on the outputfile of the previous simulation that matches the current lambda i the best (relative path)
        cord_file = os.path.relpath( f"{matching_folder}/{production}/{production}{j}.tpr", f"{new_folder}/{production}" )

        # checkpoint file pointing on the previous simulation that matches the current lambda i the best (relative path)
        cpt_file  = os.path.relpath( f"{matching_folder}/{production}/{production}{j}.cpt", f"{new_folder}/{production}" )

        # intial topology
        topo_file = os.path.relpath( initial_topo, f"{new_folder}/{production}" )

        # output file for the current lambda i
        out_file  = f"{production}{i}.tpr -maxwarn 10"

        # Define the preprocessing (grompp) & run (mdrun) GROMACS command 
        gmx_cmd_p  = f"-f {mpd_file} -c {cord_file} -p {topo_file} -t {cpt_file} -o {out_file}"
        gmx_cmd_r  = f"-deffnm {production}{i}" 

        # Define job name and output
        job_name = f"adapt_it_{iteration}_{i}"
        log_path = f"{new_folder}/LOG"

        # Rander template
        settings = { "prod": {"name": "prod", "grompp": gmx_cmd_p, "mdrun": gmx_cmd_r}, "lambda_init": i,
                     "job_name": job_name, "log_path": log_path, "working_path": new_folder}
        
        rendered = template.render( ** settings )
       
        # Create the new folder
        os.makedirs( new_folder, exist_ok = True )

        # Create job folder
        os.makedirs( f"{destination_path}/iteration_{iteration}/job_files", exist_ok = True )

        # Write new job file
        with open( f"{destination_path}/iteration_{iteration}/job_files/job_{i}.sh", "w") as f:
            f.write( rendered )


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
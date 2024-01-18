
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from pymbar import timeseries

from .free_energy_objects import ThermodynamicIntegration, FreeEnergyPerturbation

def get_simulation_data(sim_path: str, compositions: List[float], lambdas: List, both_ways: bool, free_energy_method: str="TI", delta: float=0.0001,
                        fraction: float= 0.0, density: bool=True,  density_file: str="values.sampling") -> Tuple[ ThermodynamicIntegration | FreeEnergyPerturbation, List[float]]:
    """
    Function that reads in simulation data from specified paths. This is specially designed to read in solvation free energy
    data for the insertion of a component into a mixture. Returns a free energy class object.

    Args:
        sim_path (str): Path to simulation output file. Should look like this: "**/*%.1f/**/*%d/*%d%d.*".
        compositions ( List[float]): Simulated compositions of this mixture.
        lambdas (List): Lambdas that are used in this simulation.
        both_ways (bool): If only a forward difference or a also a backward difference is performed.
        free_energy_method (str, optional): Either thermodynamic integration (TI) or Bennet acceptance ratio / free energy perturbation (BAR/FEP) method. Defaults to TI.
        delta (float, optional): Infitesimal small perturbation to compute numerical derivative. Defaults to 0.0001.
        density (bool, optional): If the mixture density should also be gathered. Defaults to True.
        fraction (float, optional): Time fraction of simulation output that should be ommited. Defaults to 0.0.
        density_file (str, optional): File name were the density values are saved. Defaults to "values.sampling".

    Returns:
        free_eng_class (ThermodynamicIntegration | FreeEnergyPerturbation): Choosen free energy object, containing the simulation data.
        dens_mix (List[float]): Mass density for each composition.
    """

    # Free energy output needed for selected free energy method for each component for each composition.
    dh_dl       = []

    # Variance of free energy output for each component for each composition.
    dh_dl_var   = []

    # Mass density for each composition
    dens_mix    = []
    
    # Loop through the mixture compositions

    for xi in compositions:

        print(f"Collect simulation data for composition: {xi}")

        paths = [ [ sim_path%(xi,i,i,i+1) ] if not both_ways else [ sim_path%(xi,i,i,i+1), sim_path%(xi,i,i,i-1) ]  for i in range( len(lambdas) ) ]

        # Read in the density for every simulated lambda for each composition and average it 
        if density:
            dens_mix.append( np.mean( [ mean_properties( os.path.dirname( path[0] )+ f"/{density_file}", keys = ["v_mass_dens"], fraction = fraction )
                                      for path in paths  ] ) )
        
        # Read in simulation results. If forward and backward path (FEP/BAR) / numeric difference (TI) is sampled
        # first read the data of forward and then of backward way
        fe_data = [ get_data( [ path[i] for path in paths ] , fraction = fraction ) for i in range( 1 if not both_ways else 2 ) ]

        # Extract the necessary free energy data
        if free_energy_method == "TI":
            dhdl, dhdlvar = get_dh_dl( fe_data, len(lambdas), delta, both_ways )

        elif free_energy_method == "BAR":
            #dhdl, dhdlvar = get_du( ... )
            pass

        else:
            raise KeyError(f"Specified free energy method is not implemented: {free_energy_method}")
        
        dh_dl.append( dhdl ) 
        dh_dl_var.append( dhdlvar )

    # Composition list for each composition (same size as the lambdas) 
    composition = [ np.ones( len(lambdas) ) * xi for xi in compositions ]

    # Lambdas used for free energy path for each component for each composition.
    lambdas     = [ np.array(lambdas) for _ in compositions ]

    # Initialize a free energy related class object with simulation data
    if free_energy_method == "TI":
        free_eng_class = ThermodynamicIntegration( lambdas = lambdas, dh_dl = dh_dl, var_dh_dl = dh_dl_var, hf_compositions = composition)

    elif free_energy_method == "BAR":
        free_eng_class = None
        
    else:
        raise KeyError(f"Specified free energy method is not implemented: {free_energy_method}")

    return free_eng_class, dens_mix

def get_data(paths: List[str], fraction: float=0.0 ) -> List[List]:
    """
    Function that reads in LAMMPS free energy output containing: Time, U1-U0, (V)exp(-(U1-U0)/kT), (V)

    Args:
        paths (List[str]): List of paths pointing to free energy output. 
        fraction (float, optional): Fraction of the time that should be disregarded. Defaults to 0.5.

    Returns:
        values (List[List]): List with sublists for every output value ( Time,dU,(V)exp(dU),(V) ) 
                             containing several numpy arrays for every intermediate.  
    """

    values = [ [],[],[],[] ]
    
    for file in paths:
        with open(file) as f:
            lines = [np.array(line.split("\n")[0].split()).astype("float") for line in f if not line.startswith("#")]
        if len(lines[0]) == 4:
            time, du, exp_du, v = [a.flatten() for a in np.hsplit( np.array(lines), 4 ) ]
        elif len(lines[0]) == 3:
            time, du, exp_du    = [a.flatten() for a in np.hsplit( np.array(lines), 3 ) ]
            v                   = np.ones(len(time))
        else:
            raise KeyError(f"Free energy file has wrong format: {file}")
        
        idx = time>fraction*max(time)

        values[0].append(time[idx])
        values[1].append(du[idx])
        values[2].append(exp_du[idx])
        values[3].append(v[idx])
    
    return values

def mean_properties(file: str, keys: List[str], fraction: float=0.5, verbose: bool=False) -> np.ndarray:
    """
    Function that reads in a LAMMPS file and return time average of given properties.
    
    Args:
        file (str): Path to LAMMPS sampling output file. 
        keys (List[str]): Variable keys to average from output (do not include timestep as it will be automaticly read in).
        fraction (float, optional): Fraction of the time that should be disregarded. Defaults to 0.5.
        verbose (bool, optional): If true, plot each read in property over the time.

    Returns:
        values (np.ndarray: Time averaged properties.
    """
    
    with open(file) as f:
        f.readline()
        keys_lmp = f.readline().split()
        idx_spec_keys = np.array([0]+[keys_lmp.index(k)-1 for k in keys if k in keys_lmp])
        lines = np.array([np.array(line.split("\n")[0].split()).astype("float")[idx_spec_keys] for line in f])

    time       = lines[:,0]
    start_time = fraction*time[-1]
    

    if verbose:
        data = [a.flatten() for a in np.hsplit(lines,lines.ndim) ]
        for i,dat in enumerate( data[1:] ):
            plt.plot(data[0][time>start_time],dat[time>start_time],label=keys[i])
            plt.ylabel(keys[i])
            plt.xlabel("Timestep")
            plt.legend()
            plt.show()
            plt.close()

    return np.squeeze( np.mean( lines[:,1:][time>start_time], axis=0 ) )

def get_dh_dl( fe_data: List[List], no_intermediates: int, delta: float, both_ways: bool=False ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function that extracts the mean and standard variance of an uncorrelated subsample of dh/dl values of a free energy
    simulation.

    Args:
        fe_data (List[ List ]): List of lists each containing simulation data (Time, <du>, <Vexp(du/rt)>, <V>) for every intermediate. 
                                If both_ways is true, two lists should be presented [data_forward, data_backward], else [data_forward]
        no_intermediates (int): No of intermediates 
        delta (float): Infitesimal small perturbation to compute numerical derivative.
        both_ways (bool, optional): If only a forward difference or a also a backward difference is performed. Defaults to False.

    Returns:
        dh_dl (np.ndarray): Mean values of dH/dl for every intermediate
        dh_dl_var (np.ndarray): Variance of the dH/dl values for every intermediate
    """


    dh_dl, dh_dl_var = [], [] 
    
    # Loop through every intermediate
    for inter in range( no_intermediates ):

        # To account for the time correlation in samples, the effective number of samples 
        # (the number of (hypothetical) independent samples required to reproduce the information content of the Nθ correlated samples) 
        # needs to be estimated. This is done by an analysis of the time autocorellation function using pymbar

        eng01   = -np.log( fe_data[0][2][inter] / fe_data[0][3][inter] ) / delta
        if both_ways: eng10   = -np.log( fe_data[1][2][inter] / fe_data[1][3][inter]) / delta * -1

        # Build central difference
        values  = ( eng01 + eng10 ) / 2 if both_ways else eng01

        try:
            # Identify an effectively uncorrelated subset of data
            indices = timeseries.subsample_correlated_data( values )
        except:
            # Take all data
            indices = np.arange(0, len(values) )

        # Get the mean and standard deviation of this subset (looked up in alchempy)
        mean    = np.mean( values[indices] )
        std     = np.std( values[indices] ) / np.sqrt( len(indices) - 1 )
        
        dh_dl.append( mean )
        dh_dl_var.append( std**2 )

    return np.array(dh_dl), np.array(dh_dl_var)
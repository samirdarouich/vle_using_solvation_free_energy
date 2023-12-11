#### Subfunctions ####

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import AutoMinorLocator
from pymbar import timeseries
from typing import List, Tuple, Dict

#### Functions that work with LAMMPS output ####

def get_simulation_data(sim_txt: str, sim_path: str, components: List[str], charged: List[bool], 
                        compositions: np.ndarray, lambdas: List, delta: float, both_ways: bool, 
                        density: bool=True, fraction: float= 0.0, density_file: str="values.sampling") -> Tuple[ List[List], List[List], List[List], List[List], np.ndarray]:
    """
    Function that reads in simulation data from specified paths. This is specially designed to read in solvation free energy
    data for a binary mixture from several independent simulations.

    Args:
        sim_txt (str): Define if van der Waals or Coulomb simulation (vdw or coul).
        sim_path (str): Path to simulation, should look like this: ".../%s/sim_%s/x%.1f/TI/sim_%s_%d/fep_%s.fep%d%d".
        components (List[str,str]): Names of the components of the mixture.
        charged (List[bool, bool]): List if the components are charged (only relevant in Coulomb case).
        compositions (np.ndarray): Simulated compositions of this mixture.
        lambdas (List): Lambdas that are used in this simulation.
        delta (float): Infitesimal small deflection to compute numerical derivative.
        both_ways (bool): If only a forward difference or a also a backward difference is performed.
        density (bool, optional): If the mixture density should also be gathered. Defaults to True.
        fraction (float, optional): Time fraction of simulation output that should be ommited. Defaults to 0.0.
        density_file (str, optional): File name were the density values are saved. Defaults to "values.sampling".

    Returns:
        l_learn (2D list): Lambdas used for free energy path for each component for each composition.
        xi_learn (2D list): Composition, for each component for each composition.
        du_dl_learn (2D list): Free energy output for each component for each composition.
        du_dl_var_l (2D list): Variance of free energy output for each component for each composition.
        dens_mix (2D array): Mass density for each composition, should be the same on both axis.
    """
    
    l_learn     = [[],[]]
    xi_learn    = [[],[]]
    du_dl_learn = [[],[]]
    du_dl_var_l = [[],[]]
    dens_mix    = np.zeros( ( 2, len(compositions) ) )
    
    ## Loop through the mixture ##

    for ni,component in enumerate( components ):
        
        # Skip uncharged component in case of Coulomb
        if sim_txt == "coulomb" and not charged[ni]: continue

        print(f"\nGather simulation data for component: {component}\n")

        ## Get data for every composition ##

        for ii,xi in enumerate( np.round( compositions, 1 ) ):

            print(f"Composition {xi}")

            # Define the mixture name
            if ni == 0:
                mixture = f"mix_{components[0]}_{components[1]}" if xi<1.0 else f"mix_{components[0]}_{components[0]}"
            else:
                mixture = f"mix_{components[1]}_{components[0]}" if xi<1.0 else f"mix_{components[1]}_{components[1]}"
            
            # If pure mixture, than xi_str is 0.0
            xi_str  = xi if xi<1.0 else 0.0

            paths = [ [ sim_path%(mixture,sim_txt,xi_str,sim_txt,i,sim_txt,i,i+1) ] if not both_ways else \
                      [ sim_path%(mixture,sim_txt,xi_str,sim_txt,i,sim_txt,i,i+1), sim_path%(mixture,sim_txt,xi_str,sim_txt,i,sim_txt,i,i-1) ]  
                      for i in range( len(lambdas) ) ]

            # Read in the density
            if density:
                tmp = [ mean_properties( os.path.dirname( path[0] )+ f"/{density_file}", keys = ["v_mass_dens"], fraction = fraction )
                        for path in paths  ]
                dens_mix[ni][ii] = np.mean( tmp )
            
            # Read in simulation results. If forward and backward path (FEP/BAR) / numeric difference (TI) is sampled
            # first read the data of forward and then of backward way
            fe_data = [ get_data( [ path[i] for path in paths ] , fraction = fraction ) for i in range( 1 if not both_ways else 2 ) ]

            # Extract the necessary free energy data
            dh_dl, dh_dl_var = get_dh_dl( fe_data, len(lambdas), delta, both_ways )

            # Add data
            l_learn[ni].append( np.array(lambdas) )
            xi_learn[ni].append( np.ones( len(lambdas) ) * xi )
            du_dl_learn[ni].append( dh_dl ) 
            du_dl_var_l[ni].append( dh_dl_var )

    return l_learn,xi_learn,du_dl_learn,du_dl_var_l,dens_mix


def get_dh_dl( fe_data: List[List], no_intermediates: int, delta: float, both_ways: bool=False ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function that extracts the mean and standard variance of an uncorrelated subsample of dh/dl values of a free energy
    simulation.

    Args:
        fe_data (List[ List ]): List containing simulation data (Time, <du>, <Vexp(du/rt)>, <V>) for every intermediate
        no_intermediates (int): No of intermediates 
        delta (float): Infitesimal small deflection to compute numerical derivative.
        both_ways (bool, optional): If only a forward difference or a also a backward difference is performed. Defaults to False.

    Returns:
        dh_dl (np.ndarray): Mean values of dH/dl for every intermediate
        dh_dl_var (np.ndarray): Variance of the dH/dl values for every intermediate
    """

    ## Loop through every intermediate ##

    dh_dl, dh_dl_var = [], [] 
    
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

def get_data(paths: List[str], adapt: bool=False, fraction: float=0.0, 
             adapt_time: float=0.0, no_intermediates: int=0 ) -> List[List]:
    """Function that reads in LAMMPS free energy output containing: Time, U1-U0, (V)exp(-(U1-U0)/kT), (V)

    Args:
        paths (List[str]): List of paths pointing to free energy output. Either just one file if adapt is used or several files if all intermediates are perfomed in lonely simulation
        adapt (bool, optional): If adapt simulation is performed.
        fraction (float, optional): Fraction of sampled data that should be evaluated
        adapt_time (float, optional): Timespan per lambda (only relevant in case of an adapt simulation).
        no_intermediates (int, optional): Number of intermediates (only relevant in case of an adapt simulation).

    Returns:
        values (List[List]): List with sublists for every output value ( Time,dU,(V)exp(dU),(V) ) 
                             containing several numpy arrays for every intermediate.  
    """

    values = [ [],[],[],[] ]
    
    # Check if one simulation with adapt is performed or several simulations
    if adapt:
        with open(paths[0]) as f:
            lines = [np.array(line.split("\n")[0].split()).astype("float") for line in f if not line.startswith("#")]
        if len(lines[0]) == 4:
            time,du,exp_du,v = [a.flatten() for a in np.hsplit( np.array(lines), 4 ) ]
        else:
            time,du,exp_du   = [a.flatten() for a in np.hsplit( np.array(lines), 3 ) ]
            v                = np.ones(len(time))
  
        start_time  = int( 0 )
        end_time    = int( no_intermediates * adapt_time )
        intervalls  = [range(i+1+int( fraction * adapt_time ), i+adapt_time+1) for i in range(start_time, end_time, adapt_time)]

        for intervall in intervalls:
            idx     = [i for i,t in enumerate(time) if int(t) in intervall]
            values[0].append(time[idx])
            values[1].append(du[idx])
            values[2].append(exp_du[idx])
            values[3].append(v[idx])
    
    else:
        for file in paths:
            with open(file) as f:
                lines = [np.array(line.split("\n")[0].split()).astype("float") for line in f if not line.startswith("#")]
            if len(lines[0]) == 4:
                time,du,exp_du,v = [a.flatten() for a in np.hsplit( np.array(lines), 4 ) ]
            else:
                time,du,exp_du   = [a.flatten() for a in np.hsplit( np.array(lines), 3 ) ]
                v                = np.ones(len(time))
            
            idx = time>fraction*max(time)

            values[0].append(time[idx])
            values[1].append(du[idx])
            values[2].append(exp_du[idx])
            values[3].append(v[idx])
    
    return values

def mean_properties(file: str, keys: List=[], fraction: float=0.7, verbose: bool=False) -> np.ndarray:
    
    """
    Function that reads in file and return time average of given properties.
    
    Args:
        file (string): Path to LAMMPS sampling output file. 
        keys (list,string): Variable keys to average from output (do not include TimeStep as it will be automaticly read in).
    
    Returns:
        values (np.ndarray,float): Time averaged properties.
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

    return np.squeeze(np.mean(lines[:,1:][time>start_time],axis=0))

def trapezoid_integration(x,y,y_var=[]):
    """
    Function that perform trapezoid integration. If provided, error propagation through integration is performed 
    and the resulting standard deviation is returned

    Args:
        x (list): Data points
        y (_type_): Data values
        y_var (list, optional): Data variance. Defaults to [].

    Returns:
        Y (float): Integral value
        Y_std (float): Standard deviation of integral value
    """
    
    weights = np.array( [(x[1]-x[0])/2] + [(x[i+1]-x[i-1])*0.5 for i,_ in enumerate(x) if not (i==0 or i==len(x)-1)] + [(x[-1]-x[-2])/2] )

    Y     = np.dot( weights, y )
    Y_std = np.sqrt( np.dot( weights**2, y_var ) ) if len(y_var)==len(x) else 0.0

    return Y, Y_std

def cubic_integration(x,y,y_var=[]):
    """
    Function that perform cubic integration. If provided, error propagation through integration is performed 
    and the resulting standard deviation is returned

    Args:
        x (list): Data points
        y (_type_): Data values
        y_var (list, optional): Data variance. Defaults to [].

    Returns:
        Y (float): Integral value
        Y_std (float): Standard deviation of integral value
    """
    
    ncs     = naturalcubicspline(np.array(x))
    weights = ncs.wk.copy()
    
    Y     = np.sum( np.dot( weights, y ) )
    Y_std = np.sqrt( np.sum( np.dot( weights**2, y_var ) ) ) if len(y_var)==len(x) else 0.0

    return Y, Y_std


def get_system_settings(name1,name2,volatile_comp,temperatures,pressures,densities):
    """
    Function that returns sublists of given equilibrium temperature, pressure, and denisties list for a mixture.
    If pure system of component 1 is under investigation, then use last entry of tempature, pressure and density
    If pure system of component 2 is under investigation, then use first entry of tempature, pressure and density
    If additions of component 1 in mixture is investigated, then use all until the penultimate entry
    If additions of component 2 in mixture is investigated, then use all except the first entry

    Component 1 is always the volatile component

    Args:
        name1 (str): Name of component 1
        name2 (str): Name of componen 2
        volatily_comp (str): Name of the volatile component
        temperatures (list): Equilibrium temperatures
        pressures (list): Equilibrium pressures
        densities (list): Equilibrium densities
    """
    
    if (name1 == name2 and name1 == volatile_comp): 
        return [ temperatures[-1] ], [ pressures[-1] ], [ densities[-1] ]
    
    elif (name1 == name2 and not name1 == volatile_comp): 
        return [ temperatures[0] ], [ pressures[0] ], [ densities[0] ]
    
    elif name1 == volatile_comp:
        return temperatures[:-1], pressures[:-1], densities[:-1]
    
    elif name2 == volatile_comp:
        return list(np.flip(temperatures)[:-1]), list(np.flip(pressures)[:-1]), list(np.flip(densities)[:-1])
    
    else:
        raise TypeError("System is not correctly specified!")

#### General functions ####

def plot_data(datas,labels,colors,path_out="",linestyle=[],markerstyle=[],ax_lim=[[],[]],ticks=[np.array([]),np.array([])],label_size=24,legend_size=18,tick_size=24,size=(8,6),lr=False,fill=[]):
    
    """
    Function that plots data
    
    Args:
        datas (list): Data list containing in each sublist x and y data (if errorbar plot is desired, datas should contain as 3rd entry the x-error and as 4th entry the y-error )
        labels (list): Labels list containing labels for each data in datas. 2nd last entry is x-axis label and last entry for y-axis
        path_out (string,optional): Path to save plot (if wanted)
        linestyle (list,optional): If any special linestyle should be specified
        markerstyle (list,optional): If any special markerstyle should be specified
        ax_lim (list of list,optional): List with ax limits
        ticks (list of np.arrays,optional): List with ticks for x and y axis
        label_size (int,optional): Size of axis labels (also influence the linewidth with factor 1/7)
        legend_size (int,optional): Size of labels in legend
        tick_size (int,optional): Size of ticks on axis
        size (touple,optional): Figure size
        lr (boolean,optional): If True then plot will have y ticks on both sides
        fill (list of booleans): If entry i is True then datas[i][1] should contrain 2 entries -> y_low,y_high

    """
    
    if not bool(fill): fill = [False for _ in datas]

    fig,ax = plt.subplots(figsize=size)

    for i,(data,label,color) in enumerate(zip(datas,labels,colors)):

        # Prevent plotting of emtpy data:
        if (not np.any(data[0]) and not isinstance(data[0],float)) or \
           (not np.any(data[1])) or \
           (not np.any(data[0]) and not np.any(data[1])): continue

        ls       = linestyle[i] if len(linestyle)>0 else "solid"
        ms       = markerstyle[i] if len(markerstyle)>0 else "."
        fill_bol = fill[i]
        error    = True if len(data) == 4 else False

        if fill_bol:
            ax.fill_between(data[0],data[1][0],data[1][1], facecolor=color,alpha=0.3)
        elif error:
            ax.errorbar(data[0],data[1],xerr=data[2],yerr=data[3],linestyle=ls,marker=ms,markersize=label_size/2,
                        linewidth=label_size/7,elinewidth=label_size/7*0.7,capsize=label_size/5,color=color,label=label )
        else:
            ax.plot(data[0],data[1],linestyle=ls,marker=ms,markersize=label_size/2,linewidth=label_size/7,color=color,label=label)

    ax.legend(fontsize=legend_size)
    
    try: 
        if len(ax_lim[0]) > 0: ax.set_xlim(*ax_lim[0])
    except: pass
    try: 
        if len(ax_lim[1]) > 0: ax.set_ylim(*ax_lim[1])
    except: pass

    for axis in ["top","bottom","left","right"]:
        ax.spines[axis].set_linewidth(2)

    if len(ticks[0])>0:
        ax.set_xticks(ticks[0])
    elif len(ticks[1])>0:
        ax.set_yticks(ticks[1])

    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="major",labelsize=tick_size,direction="out",width=tick_size/9,length=tick_size/5,right=lr)
    ax.tick_params(which="minor",labelsize=tick_size,direction="out",width=tick_size/12,length=tick_size/8,right=lr)
    ax.set_xlabel(labels[-2],fontsize=label_size)
    ax.set_ylabel(labels[-1],fontsize=label_size)
    fig.tight_layout()
    if bool(path_out): fig.savefig(path_out,dpi=400,bbox_inches='tight')
    plt.show()
    plt.close()
    
    return


def fit_poly(x,y,deg=2,w=[]):
    """
    Function that performs a polynomial fit to given data
    
    Args:
        x (list,float): Locations of data points
        y (list,float): Data for given locations
        deg (int,optional): Degree of polynomial (default: 2)
        weights (list,float,optional): List of fittings weights for each residuum (default: No weights)
    
    Returns:
        np.poly1d(a)(x_pred) (list,float): Data points of fitted polynomial to given locations
        std (list,float): Corresponding uncertanty due to polynomial fit
    
    """
    # Error propagation in polynomial least square fit (https://iopscience.iop.org/article/10.1088/0026-1394/48/1/002/pdf)
    
    X = np.array([[xi**d for d in range(deg+1)] for xi in x])

    if any(w):
        a,cov = np.polyfit(x,y,deg=deg,w=w,cov=True)
        std   = np.sqrt( np.diag( X.dot( np.flip(cov) ).dot( X.T ) ) + 1/np.array(w) )
                                                                
    else:
        a,cov = np.polyfit(x,y,deg=deg,cov=True)
        std   = np.sqrt( np.diag( X.dot( np.flip(cov) ).dot( X.T ) ) )
                                    
    return np.poly1d(a)(x),std    


def merge_nested_dicts(existing_dict, new_dict):
    for key, value in new_dict.items():
        if key in existing_dict and isinstance(existing_dict[key], dict) and isinstance(value, dict):
            # If both the existing and new values are dictionaries, merge them recursively
            merge_nested_dicts(existing_dict[key], value)
        else:
            # If the key doesn't exist in the existing dictionary or the values are not dictionaries, update the value
            existing_dict[key] = value


def work_json(file_path: str, data: Dict={}, to_do: str="read", indent: int=2):
    
    """
    Function to work with json files

    Args:
        file_path (string): Path to json file
        data (dict): If write is choosen, provide input dictionary
        to_do (string): Action to do, chose between "read", "write" and "append". Defaults to "read".

    Returns:
        data (dict): If read is choosen, returns dictionary
    """
    
    if to_do=="read":
        with open(file_path) as f:
            data = json.load(f)
        return data
    
    elif to_do=="write":
        with open(file_path,"w") as f:
            json.dump(data,f,indent=indent)

    elif to_do=="append":
        if not os.path.exists(file_path):
            with open(file_path,"w") as f:
                json.dump(data,f,indent=indent)
        else:
            with open(file_path) as f:
                current_data = json.load(f)
            merge_nested_dicts(current_data,data)
            with open(file_path,"w") as f:
                json.dump(current_data,f,indent=indent)
        
    else:
        raise KeyError("Wrong task defined: %s"%to_do)

def exception_factory(exception, message):
    "Function to raise Error"
    raise exception(message)

class naturalcubicspline():
    """
    Class that performes natural cubic spline interpolation.

    self.w_k are the weights used in an integration scheme
    """
    def __init__(self, x):

        # define some space
        L = len(x)
        H = np.zeros([L,L],float)
        M = np.zeros([L,L],float)
        BW = np.zeros([L,L],float)
        AW = np.zeros([L,L],float)
        DW = np.zeros([L,L],float)

        h = x[1:L]-x[0:L-1]
        ih = 1.0/h

        # define the H and M matrix, from p. 371 "applied numerical methods with matlab, Chapra"
        H[0,0] = 1
        H[L-1,L-1] = 1
        for i in range(1,L-1):
            H[i,i] = 2*(h[i-1]+h[i])
            H[i,i-1] = h[i-1]
            H[i,i+1] = h[i]

            M[i,i] = -3*(ih[i-1]+ih[i])
            M[i,i-1] = 3*(ih[i-1])
            M[i,i+1] = 3*(ih[i])

        CW = np.dot(np.linalg.inv(H),M)  # this is the matrix translating c to weights in f.
                                                    # each row corresponds to the weights for each c.

        # from CW, define the other coefficient matrices
        for i in range(0,L-1):
            BW[i,:]    = -(h[i]/3)*(2*CW[i,:]+CW[i+1,:])
            BW[i,i]   += -ih[i]
            BW[i,i+1] += ih[i]
            DW[i,:]    = (ih[i]/3)*(CW[i+1,:]-CW[i,:])
            AW[i,i]    = 1

        # Make copies of the arrays we'll be using in the future.
        self.x  = x.copy()
        self.AW = AW.copy()
        self.BW = BW.copy()
        self.CW = CW.copy()
        self.DW = DW.copy()

        # find the integrating weights
        self.wsum = np.zeros([L],float)
        self.wk = np.zeros([L-1,L],float)
        for k in range(0,L-1):
            w = DW[k,:]*(h[k]**4)/4.0 + CW[k,:]*(h[k]**3)/3.0 + BW[k,:]*(h[k]**2)/2.0 + AW[k,:]*(h[k])
            self.wk[k,:] = w
            self.wsum += w

    def interpolate(self,y,xnew):
        if len(self.x) != len(y):
            print("\nThe length of 'y' should be consistent with that of 'self.x'. I cannot perform linear algebra operations.")
        # get the array of actual coefficients by multiplying the coefficient matrix by the values
        a = np.dot(self.AW,y)
        b = np.dot(self.BW,y)
        c = np.dot(self.CW,y)
        d = np.dot(self.DW,y)

        N = len(xnew)
        ynew = np.zeros([N],float)
        for i in range(N):
            # Find the index of 'xnew[i]' it would have in 'self.x'.
            j = np.searchsorted(self.x, xnew[i]) - 1
            lamw = xnew[i] - self.x[j]
            ynew[i] = d[j]*lamw**3 + c[j]*lamw**2 + b[j]*lamw + a[j]
        # Preserve the terminal points.
        ynew[0] = y[0]
        ynew[-1] = y[-1]

        return ynew
    

def write_latex_table(mixture,dict_key,results,build):

    data       = work_json(results%(mixture,dict_key))
    data_build = work_json(build)

    temp        = data_build[mixture][dict_key]["temp"]
    press       = data_build[mixture][dict_key]["system_building"]["pressure"]


    comp1 = mixture.split("_")[0]
    comp2 = mixture.split("_")[1]

    # Generate LaTeX table header
    latex_table = "\\begin{tabular}{" + "".join(["c"] * 10) + "}\n"
    latex_table += "\\toprule\n"
    latex_table += "\\multicolumn{3}{c}{Fixed conditions} & \multicolumn{3}{c}{Simulation results} & \multicolumn{1}{c}{Derived properties} \\\ \n"
    latex_table += "\\cmidrule(lr){1-3} \cmidrule(lr){4-5} \cmidrule(lr){6-9} $x_\mathrm{%s}$ & $p/\mathrm{bar}$ & $T/\mathrm{K}$ & $\Delta G^\mathrm{solv.}_\mathrm{%s}/\mathrm{kJ/mol}$ &  $\Delta G^\mathrm{solv.}_\mathrm{%s}/\mathrm{kJ/mol}$ & $\gamma_\mathrm{%s}$ & $\gamma_\mathrm{%s}$ & $y_\mathrm{%s}$ & $\hat{p}/\mathrm{bar}$ \\\ \n"%(comp1,comp1,comp2,comp1,comp2,comp1)
    latex_table += "\\midrule\n"


    # jsut get even compositions
    idx = [i for i,aa in enumerate(data["x1"]) if aa in list(np.linspace(0,1,len(temp))) ]

    # Generate LaTeX table rows
    for x1, p, T, dG1, dG2,std_dG1, std_dG2, gamma1, gamma2, std_gamma1, std_gamma2, y, y_std, p_hat,p_hat_std in zip(np.array(data["x1"])[idx],press,temp,np.array(data["deltaG"][0][0])[idx],np.array(data["deltaG"][0][1])[idx],
                                                                                                                    np.array(data["deltaG"][1][0])[idx],np.array(data["deltaG"][1][1])[idx],np.array(data["gamma"][0][0])[idx],
                                                                                                                    np.array(data["gamma"][0][1])[idx],np.array(data["gamma"][1][0])[idx],np.array(data["gamma"][1][1])[idx],
                                                                                                                    np.array(data["y1"][0])[idx],np.array(data["y1"][1])[idx],np.array(data["p_equib"][0])[idx],np.array(data["p_equib"][1])[idx]):
        
        row_values = ["%.2f"%x1, "%.2f"%p, "%.2f"%T, "%.2f $\pm$ %.2f"%(dG1,std_dG1), "%.2f $\pm$ %.2f"%(dG2,std_dG2), "%.2f $\pm$ %.2f"%(gamma1,std_gamma1), "%.2f $\pm$ %.2f"%(gamma2,std_gamma2), "%.2f $\pm$ %.2f"%(y,y_std), "%.2f $\pm$ %.2f"%(p_hat,p_hat_std)]
        latex_table += " & ".join(row_values) + " \\\ \n"

    # Generate LaTeX table footer
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"

    print(latex_table)

    return
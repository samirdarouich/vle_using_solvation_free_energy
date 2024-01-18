#### Subfunctions ####

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Dict, Any
from matplotlib.ticker import AutoMinorLocator

def plot_data(datas: List[ List[List]], labels: List[str], colors: List[str],
              path_out: str="", linestyle: List[str]=[], markerstyle: List[str]=[], ax_lim: List[List[float]]=[[],[]],
              ticks: List[np.ndarray]=[np.array([]),np.array([])], label_size: int=24,
              legend_size: int=18, tick_size: int=24, size: Tuple[float]=(8,6),lr: bool=False, fill: List[bool]=[]):
    
    """
    Function that plots data.
    
    Args:
        datas (List[ List[ List, List, List, List ]]): Data list containing in each sublist x and y data 
                                                       (if errorbar plot is desired, datas should contain as 3rd entry the x-error and as 4th entry the y-error )
        labels (List[str]): Label list containing a label for each data in datas. 2nd last entry is x-axis label and last entry for y-axis
        colors (List[str]): Color list containing a color for each data in datas.
        path_out (str, optional): Path to save plot.
        linestyle (List[str], optional): Linestyle list containing a linestyle for each data in datas.
        markerstyle (List[str], optional): Markerstyle list containing a markerstyle for each data in datas.
        ax_lim (List[List[float],List[float]], optional): List with ax limits. First entry x ax, 2nd entry y ax.
        ticks (List[np.ndarray, np.ndarray] optional): List with ticks for x and y axis. First entry x ax, 2nd entry y ax.
        label_size (int,optional): Size of axis labels (also influence the linewidth with factor 1/7). Defaults to 24.
        legend_size (int,optional): Size of labels in legend. Defaults to 18.
        tick_size (int,optional): Size of ticks on axis. Defaults to 24.
        size (Tuple[float, float],optional): Figure size. Defaults to (8,6).
        lr (boolean,optional): If true, then the plot will have y ticks on both sides.
        fill (list of booleans): If entry i is True then a filled plot will be made. Therefore, datas[i][1] should contain 2 entries -> y_low,y_high instead of one -> y
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

    # Plot legend if any label provided
    if any(labels[:-2]):
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


def merge_nested_dicts(existing_dict: Dict, new_dict: Dict):
    """
    Function that merges nested dictionaries

    Args:
        existing_dict (Dict): Existing dictionary that will be merged with the new dictionary
        new_dict (Dict): New dictionary
    """
    for key, value in new_dict.items():
        if key in existing_dict and isinstance(existing_dict[key], dict) and isinstance(value, dict):
            # If both the existing and new values are dictionaries, merge them recursively
            merge_nested_dicts(existing_dict[key], value)
        else:
            # If the key doesn't exist in the existing dictionary or the values are not dictionaries, update the value
            existing_dict[key] = value


def serialize_json(data: Dict | List | np.ndarray | Any, target_class: Tuple=(), precision: int=3 ):
    """
    Function that recoursevly inspect data for classes and remove them from the data. Also convert 
    numpy arrys to lists and round floats to a given precision.

    Args:
        data (Dict | List | np.ndarray | Any): Input data.
        target_class (Tuple, optional): Class instances that should be removed from the data. Defaults to ().
        precision (int, optional): Number of decimals for floats.

    Returns:
        Dict | List | np.ndarray | Any: Input data, just without the target classes and lists instead arrays.
    """
    if isinstance(data, dict):
        return {key: serialize_json(value, target_class) for key, value in data.items() if not isinstance(value, target_class)}
    elif isinstance(data, list):
        return [serialize_json(item, target_class) for item in data]
    elif isinstance(data, np.ndarray):
        return np.round( data, precision ).tolist()
    elif isinstance(data, float):
        return round( data, precision)
    else:
        return data

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
    

from .free_energy_objects import MixtureComponent
def create_database( mixture_components: List[MixtureComponent], key: str, json_save_path: str ):
    """
    Function that takes a list of mixture component objects and drop there free energy objects as json. This can be used as low fidelity database, or just as datastorage.

    Args:
        mixture_components (List[MixtureComponent]): Mixture component objects.
        key (str): Key of free energy portion. vdw or coulomb.
        json_save_path (str): Path for the corresponding json faile
    """
    # Dump the mixture components as json.
    results = { mix_comp.component: mix_comp.free_energy_object[key].model_dump() for mix_comp in mixture_components  if key in mix_comp.free_energy_object.keys()}

    # If key is not found in both mixture components, nothing to do. 
    if not bool(results): 
        return

    mixture_key = "_".join( results.keys() )

    if np.unique( np.round( mixture_components[0].temperature, 3 ) ).size == 1:
        unique_key = int( np.unique( np.round( mixture_components[0].temperature, 0 ) )[0] )
    else:
        unique_key = int( np.unique( np.round( mixture_components[0].equilibrium_pressure, 0 ) )[0] )

    database = { mixture_key: { unique_key: results } }

    work_json( json_save_path, database, to_do="append" ) 
    
    return
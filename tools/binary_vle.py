import numpy as np
from typing import List, Tuple
from .general_utils import plot_data
from scipy.interpolate import interp1d
from .free_energy_objects import MixtureComponent


def plot_gamma_binary( component_one: MixtureComponent, component_two: MixtureComponent, save_path: str="" ):
    """
    Function that plots a binary activity coefficient diagram for two components. The first component is always the more volatile one.

    Args:
        component_one (MixtureComponent): MixtureComponent object, containing all relevant information.
        component_two (MixtureComponent): MixtureComponent object, containing all relevant information.
        save_path (str, optional): Path where the corresponding diagram should be saved. Defaults to "".
    """

    # Extract liquid compositions as well as gamma and its standard deviation
    component1 = component_one.component
    x1         = component_one.liquid_composition
    gamma1     = component_one.gamma
    gamma1_std = component_one.gamma_std

    # Flip values for component two, as everything will be plotted over the liquid composition of component one
    component2 = component_two.component
    x2         = 1 - np.array( component_two.liquid_composition )
    gamma2     = np.flip( component_two.gamma )
    gamma2_std = np.flip( component_two.gamma_std )

    # Extract reference values
    x1_ref         = component_one.gamma_composition_reference
    gamma1_ref     = component_one.gamma_reference
    
    x2_ref         = 1 - np.array( component_two.gamma_composition_reference )
    gamma2_ref     = np.flip( component_two.gamma_reference )

    reference_type = component_one.gamma_type_reference

    data   = [ [x1,gamma1,None,gamma1_std], [x2,gamma2,None,gamma2_std], [x1_ref,gamma1_ref], [x2_ref,gamma2_ref] ]
    labels = [ component1, component2, f"Reference ({reference_type})", "", "$x_\mathrm{%s}$ / -"%component1, "$\gamma$ / -" ]
    colors = [ "tab:blue", "tab:orange", "black", "black" ]
    ls     = [ "solid", "solid", "None", "None"]
    marker = [ ".", ".", "*", "*" ]

    plot_data(data,labels,colors,save_path,ax_lim=[[0.0,1.0]],linestyle=ls,markerstyle=marker,lr=True)




def compute_equilibrium_temperatures(component: str, x1: np.ndarray, temperatures: np.ndarray, y1: np.ndarray,
                                     y1_std: np.ndarray,x_ref: List=[None], y_ref: List=[None], t_ref: List=[None],
                                     save_path: str=""):
    """
    Function that visualize the T-x diagramm at constant pressure.

    Args:
        component (str): Low boiling component
        x1 (1D array): Compositions for which the free solvation energies are computed.
        tempertaures (1D array): Temperatures for which the free solvation energies are computed.
        y1 (1D array): Vapor mole fraction of component one in each composition.
        y1_std (1D array): Standard deviation for the Vapor mole fraction of component one in each composition.
        x_ref (list, optional): Liquid compositions for reference equilibrium temperatures. Defaults to [None]. 
        y_ref (list, optional): Vapor compositions for reference equilibrium temperatures. Defaults to [None].
        t_ref (list, optional): Reference equilibrium temperatures. Defaults to [None].
        save_path (str, optional): Path where to save plot. Defaults to "".
    """

    data   = [ [x1,temperatures,None,None], [y1,temperatures,None,y1_std], [x_ref,t_ref,None,None], [y_ref,t_ref,None,None] ]
    labels = [  "", "", "Reference", "", "$x_\mathrm{%s}$ / -"%component, "$T$ / K" ]
    colors = [ "tab:blue", "tab:blue",  "black", "black" ]
    ls     = [ "solid", "solid", "None", "None"]
    marker = [ ".", ".", "*", "*" ]

    plot_data(data,labels,colors,save_path,ax_lim=[[0.0,1.0]],linestyle=ls,markerstyle=marker)

    return

def compute_equilibrium_pressure(component: str, x1: np.ndarray, temperatures: np.ndarray, y1: np.ndarray,
                                 dG_mix: np.ndarray, dG_std_mix: np.ndarray, density_mix: np.ndarray,
                                 x_ref: List=[None], y_ref: List=[None], p_ref: List=[None],
                                 save_path: str="" ) -> Tuple [np.ndarray, np.ndarray]:

    """
    Function that computes the equilibrium pressure (and standard deviation) based on free solvation energy results.

    Args:
        component (str): Low boiling component
        x1 (1D array): Compositions for which the free solvation energies are computed.
        y1 (1D array): Vapor mole fraction of component one in each composition.
        tempertaure (1D array): Temperatures for which the free solvation energies are computed.
        dG_mix (2D array): Free solvation energy difference for each component in each composition.
        dG_std_mix (2D array): Standard deviation of the free solvation energy difference for each component in each composition.
        density_mix (1D array): Mass densitiy in each composition.
        x_ref (list, optional): Liquid compositions for reference equilibrium temperatures. Defaults to [None]. 
        y_ref (list, optional): Vapor compositions for reference equilibrium temperatures. Defaults to [None]. 
        y_ref (list, optional): Reference equilibrium pressures. Defaults to [None].
        save_path (str, optional): Path where to save plot. Defaults to "".

    Returns:
        p_equib (1D array): Equilibrium pressure in each composition.
        p_equib_std (1D array): Standard deviation for the equilibrium pressure in each composition.
    """

    ## Computation explained ##

    # p_i = exp( delta G_i^{solv,i+j} /(RT) ) * RT * rho_mix * x_i ##
    # p = sum_i p_i

    ## Error propagation explained ##

    # std(p_equib) = sqrt( std(p_1)**2 + std(p_2)**2 )
    # std(p_i) = p_i * beta * std(dG_mix,i)

    p_1         = np.exp( dG_mix[0] / ( 8.314*temperatures ) ) * 8.314 * temperatures * density_mix * x1
    p_2         = np.exp( dG_mix[1] / ( 8.314*temperatures ) ) * 8.314 * temperatures * density_mix * (1-x1)

    p1_std      = p_1 * 1 / ( 8.314*temperatures ) * dG_std_mix[0]
    p2_std      = p_2 * 1 / ( 8.314*temperatures ) * dG_std_mix[1]

    p_equib     = ( p_1 + p_2 ) / 10**5
    p_equib_std = np.sqrt( p1_std**2 + p2_std**2 ) /10**5

    # Compute mean realtive deviation (MRD) to reference data, if provided #
    if any(p_ref):
        p_err  = np.mean( np.abs( (interp1d(x1,p_equib,kind="cubic")(x_ref) - p_ref ) / p_ref ) ) *100
        labels = [ "", "MRD: %.2f %%"%p_err, "", "Reference" , "$x_\mathrm{%s}$, $y_\mathrm{%s}$ / -"%(component,component), "$p$ / bar" ]
    else:
        labels = [ "", "", "$x_\mathrm{%s}$, $y_\mathrm{%s}$ / -"%(component,component), "$p$ / bar" ]

    data   = [ [x1,p_equib,None,p_equib_std], [y1,p_equib,None,p_equib_std], 
               [x_ref,p_ref,None,None], [y_ref,p_ref,None,None] ]

    colors = [ "tab:blue", "tab:blue", "black", "black" ]
    ls     = [ "solid", "solid", "None", "None"]
    marker = [ "None", "None", "*", "*" ]

    plot_data(data,labels,colors,save_path,ax_lim=[[0.0,1.0]],linestyle=ls,markerstyle=marker)

    return p_equib, p_equib_std



def compute_vapor_mole_fraction(component: str, x1: np.ndarray, temperatures: np.ndarray,
                                dG_mix: np.ndarray, dG_std_mix: np.ndarray, x_ref: List=[None], y_ref: List=[None],
                                save_path: str="" ) -> Tuple [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    Function that computes the vapor mole fraction (and standard deviation), as well as 
    the seperation factor (and standard deviation) based on free solvation energy results.

    Args:
        component (str): Low boiling component
        x1 (1D array): Compositions for which the free solvation energies are computed.
        tempertaure (1D array): Temperatures for which the free solvation energies are computed.
        dG_mix (2D array): Free solvation energy difference for each component in each composition.
        dG_std_mix (2D array): Standard deviation of the free solvation energy difference for each component in each composition.
        x_ref (list, optional): Liquid compositions for reference vapor mole fractions. Defaults to [None]. 
        y_ref (list, optional): Reference vapor mole fractions for component one. Defaults to [None].
        save_path (str, optional): Path where to save plot. Defaults to "".

    Returns:
        alpha (1D array): Seperation factor in each composition.
        alpha_std (1D array): Standard deviation for the seperation factor in each composition.
        y1 (1D array): Vapor mole fraction of component one in each composition.
        y1_std (1D array): Standard deviation for the Vapor mole fraction of component one in each composition.
    """

    ## Computation explained ##

    # alpha_ij = x_i/y_i / (x_j/y_j) = np.exp( ( delta G_i^{solv,i+j} - delta G_j^{solv,i+j} ) / (RT) )
    # y_i      = x_i*alpha_ij / ( 1 + x_i * ( alpha_ij -1 ) )

    ## Error propagation explained ##
    
    # std(exp(k*u)) / exp(k*u)  = k*std(u) --> std(alpha) = alpha * beta * std( dG_mix1 - dG_mix2 )
    # std( dG_mix1 - dG_mix2 ) = sqrt( std(dG_mix1)**2 + std(dG_mix2)**2 )

    # variance of y1 (y1= x1*alpha / (1+x1*(alpha-1))): variance(y1) = variance(alpha) * (dy1/dalpha)**2
    # (dy1/dalpha) = (1-x1) * x1 / ( 1 + x1 * (alpha-1) )**2
    # std(y1) = 1-x1) * x1 / ( 1 + x1 * (alpha-1) )**2 * std(alpha)

    alpha       = np.exp( (dG_mix[0] - dG_mix[1]) / ( 8.314 * temperatures ) )

    alpha_std   = alpha * 1 / ( 8.314 * temperatures ) * np.sqrt( dG_std_mix[0]**2 + dG_std_mix[1]**2 )

    y1          = x1 * alpha / ( 1 + x1 * ( alpha - 1 ) )

    y1_std      = ( 1 - x1 ) * x1 / ( 1 + x1 * ( alpha - 1 ) )**2 * alpha_std

    # Constrain std of y1 at boundaries to be 0 #
    y1_std[np.array([0,-1])] = np.array([0.0,0.0])

    # Compute mean realtive deviation (MRD) to reference data, if provided #
    if any(y_ref):
        y_err  = np.mean( np.abs( (interp1d(x1,y1,kind="cubic")(x_ref[1:-1]) - y_ref[1:-1] ) / y_ref[1:-1] ) ) *100
        labels = [ "", "MRD: %.2f %%"%y_err, "Reference" , "$x_\mathrm{%s}$ / -"%component, "$y_\mathrm{%s}$ / -"%component ]
    else:
        labels = [ "", "", "", "$x_\mathrm{%s}$ / -"%component, "$y_\mathrm{%s}$ / -"%component ]

    data   = [ [x1,x1,None,None], [x1,y1,None,y1_std], [x_ref,y_ref,None,None] ]
    colors = [ "tab:blue", "tab:orange", "black" ]
    ls     = [ "solid", "solid", "None"]
    marker = [ "None", ".", "*" ]

    plot_data(data,labels,colors,save_path,ax_lim=[[0.0,1.0],[0.0,1.0]],linestyle=ls,markerstyle=marker)

    return alpha,alpha_std,y1,y1_std


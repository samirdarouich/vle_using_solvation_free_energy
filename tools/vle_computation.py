import numpy as np
from tools.utils import plot_data
from scipy.interpolate import interp1d
from typing import List, Tuple

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

def compute_gamma(components: List[str], x1: np.ndarray, temperatures: np.ndarray, dG_mix: np.ndarray, dG_pure: np.ndarray,
                  dG_std_mix: np.ndarray, dG_std_pure: np.ndarray, density_mix: np.ndarray, density_pure: np.ndarray,
                  x_ref: List=[None], gamma1_ref: List=[None], gamma2_ref: List=[None], save_path: str="" ) -> Tuple [np.ndarray, np.ndarray]:

    """
    Function that computes the acitivity coefficients (and standard deviation) based on free solvation energy results.

    Args:
        components (List[str]): Name of both components in the mixture.
        x1 (1D array): Compositions for which the free solvation energies are computed.
        tempertaures (1D array): Temperatures for which the free solvation energies are computed.
        dG_mix (2D array): Free solvation energy difference for each component in each composition.
        dG_pure (1D array): Free self solvation energy difference for each component.
        dG_std_mix (2D array): Standard deviation of the free solvation energy difference for each component in each composition.
        dG_std_pure (1D array): Standard deviation of the free self solvation energy difference for each component.
        density_mix (1D array): Mass density in each composition.
        density_pure (1D array): Mass density for each pure component.
        settings (dict): Dictionary containing all the necessary input
        x_ref (list, optional): Liquid compositions for reference gamma values Defaults to [None]. 
        gamma1_ref (list, optional): Reference activity coefficients for component one. Defaults to [None].
        gamma2_ref (list, optional): Reference activity coefficients for component two. Defaults to [None].

    Returns:
        gamma (2D list): Activity coefficients for each component in each composition.
        gamma_std (2D list): Standard deviation for the ativity coefficients for each component in each composition.

    """
    
    ## Computation explained ##

    # gamma_i = np.exp( ( delta G_i^{solv,i+j} - delta G_j^{solv,i+j} ) / (RT) ) * rho_{i+j} / rho_i

    ## Error propagation explained ##
    
    # Assume no error in the sampled density (reasonable since density is usually good estimated)
    # gamma = exp( (dG_mix - dG_pure) * beta)*rho_mix/rho_pure
    # std(exp(k*u)) / exp(k*u) = k*std(u) --> std(gamma) = gamma * beta * std( dG_mix - dG_pure )
    # std(dG_mix - dG_pure) = sqrt( std(dG_mix)**2 + std(dG_pure)**2 )

    gamma = [ np.exp( (dG_mix[i]-dG_pure[i]) / (8.314*temperatures) ) * density_mix / density_pure[i]  for i in range(2)]

    gamma_std = [ gamma[i] * 1/(8.314*temperatures) * np.sqrt( dG_std_mix[i]**2 + dG_std_pure[i]**2 ) for i in range(2)]

    # Constrain std of gamma at boundaries to be 0 #

    gamma_std[0][-1],gamma_std[1][0] = 0.0,0.0

    data   = [ [x1,gamma[0],None,gamma_std[0]], [x1,gamma[1],None,gamma_std[1]], [x_ref,gamma1_ref], [x_ref,gamma2_ref] ]
    labels = [ *components, "", "Reference", "$x_\mathrm{%s}$ / -"%components[0], "$\gamma$ / -" ]
    colors = [ "tab:blue", "tab:orange", "black", "black" ]
    ls     = [ "solid", "solid", "None", "None"]
    marker = [ ".", ".", "*", "*" ]

    plot_data(data,labels,colors,save_path,ax_lim=[[0.0,1.0]],linestyle=ls,markerstyle=marker,lr=True)


    return gamma,gamma_std
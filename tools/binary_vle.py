import numpy as np
from typing import List
from scipy.constants import R
from .general_utils import plot_data
from scipy.interpolate import interp1d
from .free_energy_objects import MixtureComponent

def compute_vapor_mole_fraction( component_one: MixtureComponent, component_two: MixtureComponent, save_path: str="",
                                 x_ref: np.ndarray=np.array([]), y_ref: np.ndarray=np.array([]) ):

    """
    Function that computes the vapor mole fraction (and standard deviation), as well as 
    the seperation factor (and standard deviation) based on free solvation energy results.

    The seperation factor of a binary mixture with components i and j is defined as:
        alpha_{ij} = x_i / y_i / ( x_j / y_j ) = exp( ( delta G_i^{solv,i+j} - delta G_j^{solv,i+j} ) / (RT) ) )

    From the seperation factor the vapor composition of component i can be computes as:
        y_i = x_i * alpha_{ij} / ( 1 + x_i * ( alpha_{ij} -1 ) )
        y_j = 1 - y_i (closing condition)
    
    Error propagation explained:

    with std(exp(k*u)) / exp(k*u)  = k*std(u) follows for the seperation factor:
        std(alpha_{ij}) = alpha_{ij} * beta * std( delta G_i^{solv,i+j}  - delta G_j^{solv,i+j}  )
        with std( delta G_i^{solv,i+j}  - delta G_j^{solv,i+j}  ) = sqrt( std(delta G_i^{solv,i+j})**2 + std(delta G_j^{solv,i+j})**2 )

    for the vapor composition of component i:
        variance(y_i) = var(alpha_{ij}) * (dy_i/dalpha_{ij})**2
        with (dy_i/dalpha_{ij}) = ( 1 - x_i ) * x_i / ( 1 + x_i * ( alpha_{ij} - 1 ) )**2
        
    
    Args:
        component_one (MixtureComponent): MixtureComponent object, containing all relevant information.
        component_two (MixtureComponent): MixtureComponent object, containing all relevant information.
        save_path (str, optional): Path where the corresponding diagram should be saved. Defaults to "".
        x_ref (list, optional): Liquid compositions for reference vapor mole fractions. Defaults to [None]. 
        y_ref (list, optional): Reference vapor mole fractions for component one. Defaults to [None].
    """

    # Extract liquid composition
    component1 = component_one.component
    x1         = component_one.liquid_composition

    # Extract the solvation free energy of the components into the mixture
    delta_G_solv_1_12     = component_one.solvation_free_energy
    delta_G_solv_1_12_std = component_one.solvation_free_energy_std

    # Flip values for component two, as x2 = 1 - x1
    delta_G_solv_2_12     = np.flip( component_two.solvation_free_energy )
    delta_G_solv_2_12_std = np.flip( component_two.solvation_free_energy_std )
    
    # Temperatures of both mixture should be the same
    temperature  = component_one.temperature
    
    # Compute the seperation factor
    alpha_12     = np.exp( ( delta_G_solv_1_12 - delta_G_solv_2_12 ) / ( R * temperature ) )
    alpha_12_std = alpha_12 * 1 / ( R * temperature  ) * np.sqrt( delta_G_solv_1_12_std**2 + delta_G_solv_2_12_std**2 )

    alpha_21     = np.flip( 1 / alpha_12 )
    alpha_21_std = np.flip( 1 / alpha_12**2 * alpha_12_std )

    # Compute the vapor composition of component one
    y1          = x1 * alpha_12 / ( 1 + x1 * ( alpha_12 - 1 ) )
    y1_std      = ( 1 - x1 ) * x1 / ( 1 + x1 * ( alpha_12 - 1 ) )**2 * alpha_12_std

    y2          = np.flip( 1 - y1 )
    y2_std      = np.flip( y1_std )

    # Constrain std of vapor compositions at boundaries to be 0
    y1_std[np.array([0,-1])] = 0.0, 0.0
    y2_std[np.array([0,-1])] = 0.0, 0.0

    # Add results to mixture component objects
    component_one.add_attribute( key = "seperation_factor", attribute = alpha_12, attribute_std = alpha_12_std )
    component_one.add_attribute( key = "vapor_composition", attribute = y1, attribute_std = y1_std )

    component_two.add_attribute( key = "seperation_factor", attribute = alpha_21, attribute_std = alpha_21_std )
    component_two.add_attribute( key = "vapor_composition", attribute = y2, attribute_std = y2_std )

    # Compute mean realtive deviation (MRD) to reference data, if provided
    if any(y_ref):
        y_err  = np.mean( np.abs( (interp1d(x1,y1,kind="cubic")(x_ref[1:-1]) - y_ref[1:-1] ) / y_ref[1:-1] ) ) *100
        labels = [ "", "MRD: %.2f %%"%y_err, "Reference" , "$x_\mathrm{%s}$ / -"%component1, "$y_\mathrm{%s}$ / -"%component1 ]
    else:
        labels = [ "", "", "", "$x_\mathrm{%s}$ / -"%component1, "$y_\mathrm{%s}$ / -"%component1 ]

    data   = [ [x1,x1,None,None], [x1,y1,None,y1_std], [x_ref,y_ref,None,None] ]
    colors = [ "tab:blue", "tab:orange", "black" ]
    ls     = [ "solid", "solid", "None"]
    marker = [ "None", ".", "*" ]

    plot_data(data,labels,colors,save_path,ax_lim=[[0.0,1.0],[0.0,1.0]],linestyle=ls,markerstyle=marker)

    return


from tools.free_energy_objects import MixtureComponent
from scipy.interpolate import interp1d
def compute_equilibrium_pressure( component_one: MixtureComponent, component_two: MixtureComponent, save_path: str="", computation_method: int=2,
                                  x_ref: np.ndarray=np.array([]), y_ref: np.ndarray=np.array([]), p_ref: np.ndarray=np.array([]) ):
    """
    Function that computes and plots the equilibrium pressure for a binary mixture. The first component is always the more volatile one.
    There are two different methods computing the system pressure for a binary mixture using solvation free energies:
        1.) Utilize the simplified raoult phase equilibrium relation: p = p_{i}^{sat} \gamma_{i} x_{i} / y_{i} for each component and pure vapor pressures at boundaries
        2.) Utilize the relation between partial vapor pressure and solvation free energy: p_{i} = R T \\rho_{i+j} * x_{i} * exp( delta G_i^{solv,i+j} / ( RT ) ). Where i+j denotes a mixture of i and j.
            Thus, the total pressure is given as p = p_{i} + p_{j} at matching liquid composition: xi = 1 - xj ! 
    
    Method 1.) is usuable if the provided liquid mole fractions of component two do not full fill: x1 = 1 - x2. Hence compute the system pressure individually and take the pure compoents at the boundaries.
    Method 2.) is less error sensitive, as it directly encorperates the solvation free energy and do not rely on other computed properties like gamma or the vapor mole fraction. But it is requiered that 
               the simulations are conducted at matching liquid compositions: xi = 1 - xj !

    Error propagation explained:

    Method 1.) std(p) = sqrt( var(gamma_i) * (dp/dgamma_i)**2 + var(p_sat_i) * (dp/dp_sat_i)**2 + var(y_i) * (dp/dy_i)**2 )
               with dp/dgamma_i = x_i * p_sat_i / y_i and dp/dp_sat_i = x_i * gamma_i / y_i and dp/dy_i = -x_i * gamma_i * p_sat_1 / (y_1)**2
    Method 2.) std(p) = sqrt( var(p_i) + var(p_j) ) = sqrt( std(p_i)**2 + std(p_j)**2 ) 
               with std(p_i) = p_i * std(delta G_i^{solv,i+j}) * beta with std(exp(k*u)) / exp(k*u) = k*std(u)

    Args:
        component_one (MixtureComponent): MixtureComponent object, containing all relevant information.
        component_two (MixtureComponent): MixtureComponent object, containing all relevant information.
        save_path (str, optional): Path where the corresponding diagram should be saved. Defaults to "".
        computation_method (int, optional): Which computation method should be utilized. Defaults to 2.
        x_ref (np.ndarray, optional): Numpy array with reference liquid compositions of component 1.
        y_ref (np.ndarray, optional): Numpy array with reference vapor compositions of component 1.
        p_ref (np.ndarray, optional): Numpy array with reference equilibrium pressures.
    """

    # Extract liquid and vapor compositions
    component1 = component_one.component
    x1         = component_one.liquid_composition
    y1         = component_one.vapor_composition
    
    if computation_method == 1:
        print("Computing the system pressure with: p = p_{i}^{sat} \gamma_{i} x_{i} / y_{i}\n")

        # Extract the vapor compositions and acitivity coefficients for the components
        gamma1    = component_one.gamma
        p_sat1    = component_one.pure_vapor_pressure

        x2        = component_two.liquid_composition
        y2        = component_two.vapor_composition
        gamma2    = component_two.gamma
        p_sat2    = component_two.pure_vapor_pressure

        # Compute system pressure using the properties of each component
        p_equib_1 = x1 * gamma1 * p_sat1 / y1
        p_equib_2 = x2 * gamma2 * p_sat2 / y2

        # Ensure that boundaries are the correct vapor pressures
        p_equib_1[0] = p_sat2
        p_equib_2[0] = p_sat1

        # Compute standard deviation
        dp_dgamma1 = x1 * p_sat1 / y1
        dp_dgamma2 = x2 * p_sat2 / y2

        dp_dp_sat1 = x1 * gamma1 / y1
        dp_dp_sat2 = x2 * gamma2 / y2

        dp_dy1     = -x1 * gamma1 * p_sat1 / y1**2
        dp_dy2     = -x2 * gamma2 * p_sat2 / y2**2

        gamma1_var = component_one.gamma_std**2
        gamma2_var = component_two.gamma_std**2
        
        p_sat1_var = component_one.pure_vapor_pressure_std**2
        p_sat2_var = component_two.pure_vapor_pressure_std**2
         
        y1_var     = component_one.vapor_composition_std**2
        y2_var     = component_two.vapor_composition_std**2

        p_equib_1_std = np.sqrt( gamma1_var * dp_dgamma1**2 + p_sat1_var * dp_dp_sat1**2 + y1_var * dp_dy1**2 )
        p_equib_2_std = np.sqrt( gamma2_var * dp_dgamma2**2 + p_sat2_var * dp_dp_sat2**2 + y2_var * dp_dy2**2 )

        # Ensure that boundaries are the correct vapor pressures
        p_equib_1_std[0] = np.sqrt( p_sat2_var )
        p_equib_2_std[0] = np.sqrt( p_sat1_var )

        # Add the equilibrium pressure to the mixture component objects
        component_one.add_attribute( key = "equilibrium_pressure", attribute = p_equib_1, attribute_std = p_equib_1_std )
        component_two.add_attribute( key = "equilibrium_pressure", attribute = p_equib_2, attribute_std = p_equib_2_std )

        # Adapt the composition of component 2 to plot the data over compositions of component 1
        x2            = 1 - x2
        y2            = np.flip( y2 )
        p_equib_2     = np.flip( p_equib_2 )
        p_equib_2_std = np.flip( p_equib_2_std )

        # Total system pressure concatenate both pressures
        #x1,idx      = np.unique( np.concatenate( ( x1, x2 ) ), return_index = True)
        #y1          = np.concatenate( ( y1, y2 ) )[idx]
        #p_equib     = np.concatenate( ( p_equib_1, p_equib_2 ) )[idx] / 1e5
        #p_equib_std = np.concatenate( ( p_equib_1_std, p_equib_2_std ) )[idx] / 1e5
        p_equib      = p_equib_1 / 1e5
        p_equib_std  = p_equib_1_std / 1e5
        
    elif computation_method == 2:
        print("Computing the system pressure with: p = p_{i} + p_{j} and p_{i} = R T \\rho_{i+j} * x_{i} * exp( delta G_i^{solv,i+j} / ( RT ) )\n")


        # Extract the solvation free energy of the components into the mixture
        delta_G_solv_1_12     = component_one.solvation_free_energy
        delta_G_solv_1_12_std = component_one.solvation_free_energy_std

        # Flip values for component two, as everything will be plotted over the liquid composition of component one
        x2                    = 1 - component_two.liquid_composition
        delta_G_solv_2_12     = np.flip( component_two.solvation_free_energy )
        delta_G_solv_2_12_std = np.flip( component_two.solvation_free_energy_std )

        # x2 should match the closing condition with x1
        if not all( x1 + x2 == 1 ):
            raise ValueError("Provided liquid compositions do not match!\n")

        # Temperatures of both mixture should be the same
        temperature       = component_one.temperature

        # Densities of both mixtures should be the same ( just in case average from both components )
        mixture_density   = np.mean( [ component_one.molar_density, np.flip( component_two.molar_density ) ], axis = 0 )

        # Compute the partial pressures in each statepoint (out of perspective of component 1)
        p_1         = np.exp( delta_G_solv_1_12 / ( R * temperature ) ) * R * temperature * mixture_density * x1
        p_2         = np.exp( delta_G_solv_2_12 / ( R * temperature ) ) * R * temperature * mixture_density * x2

        # Compute the standard deviation of the partial pressures in each statepoint (out of perspective of component 1)
        p1_std      = p_1 * 1 / ( R * temperature ) * delta_G_solv_1_12_std
        p2_std      = p_2 * 1 / ( R * temperature ) * delta_G_solv_2_12_std

        p_equib     = ( p_1 + p_2 ) / 1e5
        p_equib_std = np.sqrt( p1_std**2 + p2_std**2 ) / 1e5

        # Add the equilibrium pressure to the mixture component objects
        component_one.add_attribute( key = "equilibrium_pressure", attribute = p_equib * 1e5, attribute_std = p_equib_std * 1e5  )
        component_two.add_attribute( key = "equilibrium_pressure", attribute = np.flip( p_equib * 1e5 ), attribute_std = np.flip( p_equib_std * 1e5 ) )

    else:
        raise KeyError(f"Wrong computation method choosen: {computation_method}! Valid are 1 or 2.\n")

    # Plot the p-x diagram 

    # Compute mean realtive deviation (MRD) to reference data, if provided
    if any(p_ref):
        p_err  = np.mean( np.abs( ( interp1d( x1, p_equib, kind="cubic" )( x_ref ) - p_ref ) / p_ref ) ) *100
        labels = [ "", "MRD: %.2f %%"%p_err, "", "Reference" , "$x_\mathrm{%s}$, $y_\mathrm{%s}$ / -"%(component1,component1), "$p$ / bar" ]
    else:
        labels = [ "", "", "$x_\mathrm{%s}$, $y_\mathrm{%s}$ / -"%(component1,component1), "$p$ / bar" ]

    data   = [ [x1,p_equib,None,p_equib_std], [y1,p_equib,None,p_equib_std], 
               [x_ref,p_ref,None,None], [y_ref,p_ref,None,None] ]

    colors = [ "tab:blue", "tab:blue", "black", "black" ]
    ls     = [ "solid", "solid", "None", "None"]
    marker = [ "None", "None", "*", "*" ]

    plot_data( data, labels, colors, save_path, ax_lim=[[0.0,1.0]], linestyle=ls, markerstyle=marker )

    return

def plot_equilibrium_temperatures( component_one: MixtureComponent, component_two: MixtureComponent, save_path: str="",
                                   x_ref: List=[None], y_ref: List=[None], t_ref: List=[None] ):
    """
    Function that visualize the T-x diagramm at constant pressure.

    Args:
        component_one (MixtureComponent): MixtureComponent object, containing all relevant information.
        component_two (MixtureComponent): MixtureComponent object, containing all relevant information.
        save_path (str, optional): Path where the corresponding diagram should be saved. Defaults to "".
        x_ref (list, optional): Liquid compositions for reference equilibrium temperatures. Defaults to [None]. 
        y_ref (list, optional): Vapor compositions for reference equilibrium temperatures. Defaults to [None].
        t_ref (list, optional): Reference equilibrium temperatures. Defaults to [None].
        save_path (str, optional): Path where to save plot. Defaults to "".
    """

    # Extract liquid and vapor compositions
    component1   = component_one.component
    x1           = component_one.liquid_composition
    y1           = component_one.vapor_composition
    y1_std       = component_one.vapor_composition_std
    temperature1 = component_one.temperature

    # Adapt the composition of component 2 to plot the data over compositions of component 1
    x2           = 1 - component_two.liquid_composition
    y2           = component_two.vapor_composition
    y2_std       = component_two.vapor_composition_std
    temperature2 = component_two.temperature

    # Concatenate both components
    x1           = np.concatenate( ( x1, x2 ) )
    y1           = np.concatenate( ( y1, y2 ) )
    y1_std       = np.concatenate( ( y1_std, y2_std ) )
    temperatures = np.concatenate( ( temperature1, temperature2 ) )

    data   = [ [x1,temperatures,None,None], [y1,temperatures,None,y1_std], [x_ref,t_ref,None,None], [y_ref,t_ref,None,None] ]
    labels = [  "", "", "Reference", "", "$x_\mathrm{%s}$ / -"%component1, "$T$ / K" ]
    colors = [ "tab:blue", "tab:blue",  "black", "black" ]
    ls     = [ "solid", "solid", "None", "None"]
    marker = [ ".", ".", "*", "*" ]

    plot_data(data,labels,colors,save_path,ax_lim=[[0.0,1.0]],linestyle=ls,markerstyle=marker)

    return

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

    # Use closing condition to adjust x2, as the plot is done over x1
    component2 = component_two.component
    x2         = 1 - np.array( component_two.liquid_composition )
    gamma2     = component_two.gamma
    gamma2_std = component_two.gamma_std

    # Extract reference values
    x1_ref         = component_one.gamma_composition_reference
    gamma1_ref     = component_one.gamma_reference
    
    x2_ref         = 1 - np.array( component_two.gamma_composition_reference )
    gamma2_ref     = component_two.gamma_reference

    reference_type = component_one.gamma_type_reference

    data   = [ [x1,gamma1,None,gamma1_std], [x2,gamma2,None,gamma2_std], [x1_ref,gamma1_ref], [x2_ref,gamma2_ref] ]
    labels = [ component1, component2, f"Reference ({reference_type})", "", "$x_\mathrm{%s}$ / -"%component1, "$\gamma$ / -" ]
    colors = [ "tab:blue", "tab:orange", "black", "black" ]
    ls     = [ "solid", "solid", "None", "None"]
    marker = [ ".", ".", "*", "*" ]

    plot_data(data,labels,colors,save_path,ax_lim=[[0.0,1.0]],linestyle=ls,markerstyle=marker,lr=True)

    return

def plot_solvation_free_energy_binary( component_one: MixtureComponent, component_two: MixtureComponent, save_path: str=""):
    """
    Function that plots the solvation free energies for a binary mixture. The first component is always the more volatile one.

    Args:
        component_one (MixtureComponent): MixtureComponent object, containing all relevant information.
        component_two (MixtureComponent): MixtureComponent object, containing all relevant information.
        save_paths (list[str], optional): Path where the corresponding diagram (vdw or coulomb contribution) should be saved. Defaults to "". 
    """

    flag      = False
    flag_path = bool(save_path)
    charged   = [ False, False ]

    for i, key in enumerate([ "vdw", "coulomb" ]):
        
        plot_key = "vdW" if i == 0 else "coul"
        
        # Extract liquid compositions as well as solvation free energies and its standard deviation
        component1            = component_one.component
        x1                    = component_one.liquid_composition

        try:
            delta_G_solv_1_12     = component_one.solvation_free_energy_contributions[key]["delta_G"]
            delta_G_solv_1_12_std = np.sqrt(component_one.solvation_free_energy_contributions[key]["var_delta_G"])
            
            # If Coulomb entries exists, this component is charged
            if i == 1: charged[0] = True
        except:
            delta_G_solv_1_12     = np.zeros(1)
            delta_G_solv_1_12_std = np.zeros(1)

        # Adapt the composition of component 2 to plot the data over compositions of component 1
        component2            = component_two.component
        x2                    = 1 - component_two.liquid_composition

        try:
            delta_G_solv_2_12     = component_two.solvation_free_energy_contributions[key]["delta_G"]
            delta_G_solv_2_12_std = np.sqrt(component_two.solvation_free_energy_contributions[key]["var_delta_G"])

            # If Coulomb entries exists, this component is charged
            if i == 1: charged[1] = True
        except:
            delta_G_solv_2_12     = np.zeros(1)
            delta_G_solv_2_12_std = np.zeros(1)
        
        # If no component is charged, skipp the coulomb and the total plot (which equals the vdW plot in this case)
        if not any(charged):
            continue

        data   = [ [x1,delta_G_solv_1_12/1000], [x2,delta_G_solv_2_12/1000],
                   [x1,[ (delta_G_solv_1_12-1.*delta_G_solv_1_12_std)/1000, (delta_G_solv_1_12+1.*delta_G_solv_1_12_std)/1000] ],
                   [x2,[ (delta_G_solv_2_12-1.*delta_G_solv_2_12_std)/1000, (delta_G_solv_2_12+1.*delta_G_solv_2_12_std)/1000] ],
                   [x1[-1],delta_G_solv_1_12[-1]/1000,None,delta_G_solv_1_12_std[-1]/1000], [x2[0],delta_G_solv_2_12[0]/1000,None,delta_G_solv_2_12_std[0]/1000] ]

        labels = [ "$\Delta G_\mathrm{%s}^\mathrm{mix}$"%component1, 
                   "$\Delta G_\mathrm{%s}^\mathrm{mix}$"%component2,
                   "",
                   "",
                   "$\Delta G_\mathrm{%s}^\mathrm{pure}$"%component1, 
                   "$\Delta G_\mathrm{%s}^\mathrm{pure}$"%component2, 
                   "$x_\mathrm{%s}$ / -"%component1, "$\Delta G^\mathrm{solv., %s}$ / kJ/mol"%plot_key ]
            
        colors = [ "tab:blue", "tab:orange", "tab:blue", "tab:orange", "tab:blue", "tab:orange" ]
        ls     = [ "solid", "solid", "None", "None", "None", "None"]
        marker = [ "None","None", "None","None", "x", "x" ]
        
        fill   = [False, False, True, True, False, False]

        path_out = save_path%plot_key if flag_path else ""

        plot_data(data,labels,colors,path_out=path_out,ax_lim=[[0.0,1.0]],linestyle=ls,markerstyle=marker,lr=True,fill=fill)

        if i == 1: flag = True
    
    if flag:
        
        # Plot total solvation free energy

        plot_key = "total"
        
        # Extract liquid compositions as well as solvation free energies and its standard deviation
        component1            = component_one.component
        x1                    = component_one.liquid_composition
        delta_G_solv_1_12     = component_one.solvation_free_energy
        delta_G_solv_1_12_std = component_one.solvation_free_energy_std

        # Adapt the composition of component 2 to plot the data over compositions of component 1
        component2            = component_two.component
        x2                    = 1 - component_two.liquid_composition
        delta_G_solv_2_12     = component_two.solvation_free_energy
        delta_G_solv_2_12_std = component_two.solvation_free_energy_std
        
        data   = [ [x1,delta_G_solv_1_12/1000], [x2,delta_G_solv_2_12/1000],
                   [x1,[ (delta_G_solv_1_12-1.*delta_G_solv_1_12_std)/1000, (delta_G_solv_1_12+1.*delta_G_solv_1_12_std)/1000] ],
                   [x2,[ (delta_G_solv_2_12-1.*delta_G_solv_2_12_std)/1000, (delta_G_solv_2_12+1.*delta_G_solv_2_12_std)/1000] ],
                   [x1[-1],delta_G_solv_1_12[-1]/1000,None,delta_G_solv_1_12_std[-1]/1000], [x2[0],delta_G_solv_2_12[0]/1000,None,delta_G_solv_2_12_std[0]/1000] ]

        labels = [ "$\Delta G_\mathrm{%s}^\mathrm{mix}$"%component1, 
                   "$\Delta G_\mathrm{%s}^\mathrm{mix}$"%component2,
                   "",
                   "",
                   "$\Delta G_\mathrm{%s}^\mathrm{pure}$"%component1, 
                   "$\Delta G_\mathrm{%s}^\mathrm{pure}$"%component2, 
                   "$x_\mathrm{%s}$ / -"%component1, "$\Delta G^\mathrm{solv., %s}$ / kJ/mol"%plot_key ]
            
        colors = [ "tab:blue", "tab:orange", "tab:blue", "tab:orange", "tab:blue", "tab:orange" ]
        ls     = [ "solid", "solid", "None", "None", "None", "None"]
        marker = [ "None","None", "None","None", "x", "x" ]
        
        fill   = [False, False, True, True, False, False]

        path_out = save_path%plot_key if flag_path else ""

        plot_data(data,labels,colors,path_out=path_out,ax_lim=[[0.0,1.0]],linestyle=ls,markerstyle=marker,lr=True,fill=fill)

    return
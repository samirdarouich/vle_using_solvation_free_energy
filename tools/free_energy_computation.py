import numpy as np
from typing import List, Tuple, Dict
from scipy.constants import R
from scipy.interpolate import interp1d
from .numeric_utils import fit_poly
from .reader import get_simulation_data
from .free_energy_objects import ThermodynamicIntegration, FreeEnergyPerturbation, MixtureComponent
from .multi_fidelity import get_lf_training_data, prep_mf_input, MF

def get_mixture_components( molecule_name_list: List[str], coupling_lambdas: List[List[float]], main_path: str, Mol_masses: List[float], 
                            x_pred: np.ndarray=np.linspace(0,1,21), delta: float=0.001, both_ways: bool=True, free_eng_method: str="TI", free_eng_sub_style: str="",
                            integration_method: str="cubicspline", verbose: bool=False,
                            thermodynamic_settings_dict: Dict[str, Dict[str, List]]={"molecule1": {"composition": [], "temperature": [], "activity_coeff": []} },
                            multifidelity_settings_dict: Dict[str, List]={"lf_databanks":[], "lf_mixtures": [], "lf_unique_key": [], "lengthscales":[], "fix_lengthscale":False, "fix_hf_noise": True}, 
                            ):

    """
    Function that reads in simulation data from a specified simulation path (molecule_name/liquid_compositiom/coupling_key/free_energy_method/sim_*/fep*.sampling) to get the 
    solvation free energy of the component in a given mixture. For each component a dictionary containing the thermodynamic settings (compositions simulated, 
    temperatures and reference activity coeffients) should be provided. Furthermore, if thermodynamic integration with multi fidelity modeling should be utilized, a dictionary,
    containing all relevant settings (Path to low fidelity databank, low fidelity mixture, thermodynamic key of the low fidelity mixture, lengthscale hyperparameters, 
    a boolen if they should be fixed or optimized, and a boolean deciding if the simulation noise should be used to fix the high fidelity model noise) should be provided.
    
    Args:
        molecule_name_list (List[str]): List of all components that are coupled. 
        coupling_lambdas (List[List[float]]): List with coupling lambdas used for every component. In each list, coupling lambdas for the van der Waals, as well as the Coulomb coupling
                                              should be provided.
        main_path (str): Main path to the simulation folder, the following path to simulation results should look like this: molecule_name/liquid_compositiom/coupling_key/free_energy_method/sim_*/fep*.sampling
        Mol_masses (List[float]): List with molar masses of every component in the mixture (given in kg/mol).
        x_pred (np.ndarray, optional): Liquid composition array for which the solvation free energies, densities and simulation temperatures will be fitted/interpolated for. 
                                       Defaults to np.ndarray=np.linspace(0,1,21).
        delta (float, optional): Perturbation in case of thermodynamic integration, this is needed to compute the numerical derivative of the potential energy / enthalpy with respect to lambda.
                                 Defaults to 0.001.
        both_way (bool, optional): If two perturbations are performed. E.g: TI: forward and backward difference, FEP: calculation to the previous and next lambda. 
                                   Or only in forward direction if false. Defaults to True.
        free_eng_method (str, optional): Free energy method used. Defaults to TI.
        free_eng_sub_style (str, optional): Sub style of the free energy method. E.g.: for TI (multi fidelity modeling) or for FEP (BAR, FEP forward, FEP backward). Defaults to "".
        integration_method (str, optional): Integration method for thermodynamic integration. Trapezoid scheme (trapezoid), Cubic splines (cubicspline) 
                                            or a Gaussian process regression (GRP). Defaults to cubicspline.
        verbose (str, optional): If more output should be print to screen. Defaults to false.
        thermodynamic_settings_dict (Dict[str, Dict[str, List]]): Dictionary containing all necessary thermodynamic information for each component. 
                                                                  Defaults to {"molecule1": {"composition": [], "temperature": [], "activity_coeff": []} }.
        multifidelity_settings_dict (Dict[str, List]): Dictionary containing all necessary multi fidelity settings.
                                                       Defaults to {"lf_databanks":[], "lf_mixtures": [], "lf_unique_key": [], "lengthscales":[], "fix_lengthscale":False, "fix_hf_noise": True}.
    """
    
    mixture_components = []

    for i, molecule_name in enumerate( molecule_name_list):
        print(f"\nInsertion of {molecule_name}\n")

        # Extract simulated thermodynamic settings
        composition    = thermodynamic_settings_dict[molecule_name]["composition"]
        temperature    = thermodynamic_settings_dict[molecule_name]["temperatures"]
        activity_coeff = thermodynamic_settings_dict[molecule_name]["activity_coeff"]

        # Interpolate the temperature to match the new liquid compositions
        interpolation_method = "linear" if len(composition) <= 2 else "quadratic" if len(composition) == 3 else "cubic"
        temperature          = interp1d( composition, temperature, kind = interpolation_method )( x_pred )

        # Gather simulation data and compute the solvation free energy
        mass_density = []
        delta_G_contributions = {}
        free_energy_class_contributions = {}

        for j,lambdas in enumerate(coupling_lambdas[i]):
            if not lambdas: continue

            # Define type of coupled interaction
            coupling_key = "vdw" if j == 0 else "coulomb"

            print(f"\nAcquire data for {coupling_key} contribution\n")

            # Define simulation path
            sim_path = f"{main_path}/{molecule_name}_coupled/x%.1f/{coupling_key}/{free_eng_method}/sim_%d/fep%d%d.sampling"

            free_eng_class, mass_dens = get_simulation_data( sim_path = sim_path, compositions = composition, lambdas = lambdas, both_ways = both_ways,
                                                             free_energy_method = free_eng_method, delta = delta )
            
            # LAMMPS output in g/cm^3 -> convert in kg/m^3
            mass_density.append( np.array( mass_dens ) * 1000 )

            # Get the resulting free energy difference using the specified free energy method
            settings_dict = { "free_eng_class": free_eng_class, "component": molecule_name, "free_energy_method": free_eng_method, "x_pred": x_pred,
                              "integration_method": integration_method, "free_eng_sub_style": free_eng_sub_style, "verbose": verbose,
                              "lf_databank": multifidelity_settings_dict["lf_databanks"][j], "lf_mixture": multifidelity_settings_dict["lf_mixtures"][j], 
                              "lf_unique_key": multifidelity_settings_dict["lf_unique_keys"][j], "lf_component": multifidelity_settings_dict["lf_components"][j][i],
                              "lengthscale": multifidelity_settings_dict["lengthscales"][j], "fix_lengthscale": multifidelity_settings_dict["fix_lengthscale"],
                              "fix_hf_noise": multifidelity_settings_dict["fix_hf_noise"] }
            
            delta_G, var_delta_G = get_delta_G( **settings_dict )

            # Save the solvation free energy for each contribution (convert from dimensionless (G/(RT)) to J/mol)
            delta_G_contributions[coupling_key] = { "delta_G": delta_G * R * temperature, "var_delta_G": var_delta_G * ( R * temperature )**2 }

            # Save the free energy class
            free_energy_class_contributions[coupling_key] = free_eng_class
        
        # Average the mass density and interpolate it to match the new liquid compositions
        mass_density         = interp1d( composition, np.mean(mass_density, axis=0), kind = interpolation_method )( x_pred )

        # Define the molecular weight of the mixture based on the composition
        molecular_mass       = [ np.dot( Mol_masses, [ x, 1 - x ] if i == 0 else [ 1 - x, x ])  for x in x_pred ]
        
        # Save all data in the mixture component class
        settings_dict = { "component": molecule_name, "liquid_composition": x_pred, "temperature": temperature, "mass_density": mass_density,
                        "solvation_free_energy_contributions": delta_G_contributions, "molecular_mass": molecular_mass }
        
        mixture_component = MixtureComponent( **settings_dict )

        # Add reference gammas from thermodynamic input
        mixture_component.add_reference_gamma( reference_composition = composition, reference_gamma = activity_coeff )
        
        # Add free energy objects
        for key,item in free_energy_class_contributions.items():   
            mixture_component.add_free_energy_object( key = key, free_energy_object = item )

        mixture_components.append( mixture_component )

    return mixture_components

def get_delta_G( free_eng_class: ThermodynamicIntegration | FreeEnergyPerturbation, component: str, free_energy_method: str, x_pred: np.ndarray, free_eng_sub_style: str="",
                 integration_method: str="cubicspline", poly_degree: int=3, lf_databank: str="", lf_mixture: str="", lf_unique_key: str="", lf_component: str="",
                 lengthscale: List[float]=[], fix_lengthscale: bool=False, fix_hf_noise: bool=True, verbose: bool=False ) -> Tuple[ List[float], List[float] ]:
    """
    Function that uses a free energy class object and obtain the free energy difference over a given composition. 
    Possibilites are thermodynamic integration, exponential averaging or the BAR method.

    Args:
        free_eng_class (ThermodynamicIntegration | FreeEnergyPerturbation)
        component (str): Names of the component.
        free_energy_method (str): Method for evaluation of the the free energy difference.
        x_pred (1D array): New evaluated liquid compositions.
        free_eng_sub_style (str, optional): Sub style of free energy method. E.g.: multi fidelity, FEP, BAR, ...
        integration_method (str, optional): Integration method. Defaultst to cubicspline.
        poly_degree (int, optional): Degree of the polynomial fit that is performed on the sampled free energies over the composition (if 2d multifidelity, this is not the case).
        lf_databank (str, optional): Path to low fidelity databank (for vdW / Coulomb ).
        lf_mixtured (str, optional): Mixture that should be utilized as low fidelity model (named: component1_component2) (for vdW / Coulomb ).
        lf_unique_key (str, optional): Thermodynamic key for this mixture (e.g.: Temperature or pressure) (for vdW / Coulomb ).
        lengthscales (List[float], optional): Possible default lenghtscale hyperparameters (for vdW / Coulomb ). Defaults to [].
        fix_lengthscale (bool, optional): If the lenghtscale hyperparameters should be fixed. Defaults to False.
        fix_hf_noise (bool, optional): If the noise of the high fidelity should be fixed. Defaults to True.
        verbose (bool, optional): If detailed information should be printed out. Defaults to False.

    Returns:
        delta_G (List[float]): Free energy difference for each component in each composition.
        delta_G_var (List[float]): Standard deviation of the free energy difference for each component in each composition.

    """

    if free_energy_method == "TI":
        
        # This dimension defines if the multifidelit or GPR approaches should be used 1D (Interpolate for each composition only the dh/dlambda curve) or
        # if a 2D interpolation should be done, so dh/dlambda over lambda and over the composition. The 2D case avoids the need for polynomial fitting
        # of the free solvation energies via the composition.
        dimension = 2 if "2d" in free_eng_sub_style else 1

        if "multi_fidelity" in free_eng_sub_style:
            # Perform integration on 2D or 1D multifidelity simulation data (first dimension: lambda, second dimension: composition)

            # Get low fidelity data and prepare mf data input
            settings_dict = {"lf_databank": lf_databank, "lf_mixture": lf_mixture, "lf_unique_key": lf_unique_key, "lf_component": lf_component }

            lf_lambdas, lf_compositions, lf_dh_dl = get_lf_training_data( **settings_dict )
            
            settings_dict = { "component": component, "lf_compositions": lf_compositions, "lf_lambdas": lf_lambdas, "lf_dh_dl": lf_dh_dl,
                              "dimension": dimension, "x_pred": x_pred, "lengthscale": lengthscale, "fix_lengthscale": fix_lengthscale,
                              "fix_hf_noise": fix_hf_noise, "verbose": verbose}
            
            free_eng_class.multifidelity( **settings_dict  )

            # Integrate the high fidelity data
            delta_G, delta_G_var = free_eng_class.integrate( integration_method = integration_method )

            # If 1d multi fidelity is performed, fit a polynomial to the delta G's 
            if dimension == 1: 
                delta_G, delta_G_std = fit_poly( free_eng_class.compositions, delta_G, x_pred, deg=poly_degree, w=1/np.array(delta_G_var) )
                delta_G_var          = delta_G_std**2

        elif "gpr" in free_eng_sub_style:
            # Perform integration on 2D or 1D GPR simulation data (first dimension: lambda, second dimension: composition)

            # Prepare necessary input
            settings_dict = { "component": component,"dimension": dimension, "x_pred": x_pred, 
                              "lengthscale": lengthscale, "fix_lengthscale": fix_lengthscale,
                              "fix_hf_noise": fix_hf_noise, "verbose": verbose}
            
            free_eng_class.gpr( **settings_dict  )

            # Integrate the high fidelity data
            delta_G, delta_G_var = free_eng_class.integrate( integration_method = integration_method )

            # If 1d multi fidelity is performed, fit a polynomial to the delta G's 
            if dimension == 1: 
                delta_G, delta_G_std = fit_poly( free_eng_class.compositions, delta_G, x_pred, deg=poly_degree, w=1/np.array(delta_G_var) )
                delta_G_var          = delta_G_std**2

        else:
            # Integrate the raw data and fit a polynomial to it to match it to the predicted compositon
            delta_G, delta_G_var = free_eng_class.integrate( integration_method = integration_method )
            delta_G, delta_G_std = fit_poly( np.unique( free_eng_class.compositions ), delta_G, x_pred, deg=poly_degree, w=1/np.array(delta_G_var) )
            delta_G_var          = delta_G_std**2
            
        if verbose:
            if dimension == 2:
                settings_dict = { "plot_3d":True, "labels": ["x$_\mathrm{%s}$"%component,
                                                              "$\lambda$", 
                                                              "$ \\langle \\frac{\partial U}{\partial \lambda} \\rangle_{\lambda} \ / \ (k_\mathrm{B}T)$"]  }
            else:
                settings_dict = {}
            free_eng_class.plot( **settings_dict )
    
    return delta_G, delta_G_var



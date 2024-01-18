
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import R
from typing import List, Dict, Tuple
from matplotlib.ticker import AutoMinorLocator

from .general_utils import plot_data, serialize_json
from .numeric_utils import cubic_integration, trapezoid_integration
from .multi_fidelity import get_hf_prediction as get_hf_prediction_mf
from .gaussian_process import get_hf_prediction as get_hf_prediction_gpr

class ThermodynamicIntegration:
    def __init__(self, lambdas: List[List[float]], dh_dl: List[List[float]], var_dh_dl: List[List[float]]=None, hf_compositions: List[List[float]]=None):
        """
        Class that uses the derivative of the potential energy / enthaply with respect to a coupling parameter lambda
        and performs thermodynamic integration using different integration methods.

        Args:
            lambdas (List[List[float]]): List of coupling lambda lists along the integration should be conducted.
            dh_dl (List[List[float]]): List of derivative lists of the potential energy / enthaply with respect to a coupling parameter lambda.
            var_dh_dl (List[List[float]], optional): Lists of variance lists of the derivative of the potential energy / 
                                                     enthaply with respect to a coupling parameter lambda. Defaults to None.
            hf_compositions (List[List[float]]): Compositions corresponding to the the high fidelity lambda points. Only necessary for multi fidelity modeling.
                                                 (each composition list should be the same length as each lambda sublist). Defaults to None
        """
        self.lambdas            = lambdas
        self.dh_dl              = dh_dl
        self.var_dh_dl          = [[] for _ in range(len(lambdas))] if var_dh_dl == None else var_dh_dl
        self.compositions       = [0.0 for _ in range(len(lambdas))] if hf_compositions == None else hf_compositions

        # Save the original sampling data, in case a GPR or multifidelity modeling is used, which  will overwrite the original attributes
        self.overwriten           = False
        self.sampled_compositions = self.compositions.copy()
        self.sampled_lambdas      = self.lambdas.copy()
        self.sampled_dh_dl        = self.dh_dl.copy()
        self.sampled_var_dh_dl    = self.var_dh_dl.copy()


    def integrate(self, integration_method: str="cubicspline") -> Tuple[np.ndarray, np.ndarray]:
        """
        Function that performes the integration with specified method.

        Args:
            integration_method (str, optional): Which kind of integration method should be utilized. Defaults to "cubicspline".

        Returns:
            dG (np.ndarray): Free energy difference.
            var_dG (np.ndarray): Variance of the free energy difference.
        """
        self.dG     = []
        self.var_dG = []

        for lambdas, dh_dl, var_dh_dl in zip( self.lambdas, self.dh_dl, self.var_dh_dl ):

            if integration_method == "cubicspline":
                dG, var_dG = cubic_integration( x = lambdas, y = dh_dl, y_var = var_dh_dl )
            elif integration_method == "trapezoid":
                dG, var_dG = trapezoid_integration( x = lambdas, y = dh_dl, y_var = var_dh_dl )
            else:
                raise KeyError(f"Integration scheme not implemented: {integration_method}")

            self.dG.append(dG)
            self.var_dG.append(var_dG)

        return np.array(self.dG), np.array(self.var_dG)
    
    def multifidelity(self, component: str, lf_compositions: List[List[float]], lf_lambdas: List[List[float]], lf_dh_dl: List[List[float]], 
                      lf_var_dh_dl: List[List[float]]=None, dimension: int=2, x_pred: np.ndarray=np.array([]), lengthscale: List[float]=[], 
                      fix_lengthscale: bool=False, fix_hf_noise: bool=True, verbose: bool=False):
        """
        Function that uses multifidelity GPR modeling to interpolate high fidelity data.

        Args:
            component (str): Component for which the multi fidelity modeling is performed.
            lf_compositions (List[List[float]]): Compositions corresponding to the the low fidelity lambda points (each composition list should be the same length as each lambda sublist)
            lf_lambdas (List[List[float]]): Low fidelity model of list of coupling lambda lists along the integration should be conducted.
            lf_dh_dl (List[List[float]]): Low fidelity model of list of derivative lists of the potential energy / enthaply with respect to a coupling parameter lambda.
            lf_var_dh_dl (List[List[float]], optional): Low fidelity model of lists of variance lists of the derivative of the potential energy / 
                                                        enthaply with respect to a coupling parameter lambda. Defaults to None.
            dimension (int, optional): Dimension of the multi fidelity modeling. 2D includes the compositions, 1D is just dH/dl over lambda. Defaults to 2.
            x_pred (np.ndarray): New evaluated liquid compositions (in case of 2D multi fidelity). Defaults to np.array([]).
            lengthscale (List[float], optional): Possible default lenghtscale hyperparameters. Defaults to [].
            fix_lengthscale (bool, optional): If the lenghtscale hyperparameters should be fixed. Defaults to False.
            fix_hf_noise (bool, optional): If the noise of the high fidelity should be fixed. Defaults to True.
            verbose (bool, optional): If detailed information should be printed out. Defaults to False.                                            
        """

        # Call the multifidelity class to obtain interpolated high fidelity results
        hf_data = [ self.lambdas, self.compositions, self.dh_dl, self.var_dh_dl ]
        lf_data = [ lf_lambdas, lf_compositions, lf_dh_dl, lf_var_dh_dl ]
        
        settings_dict = { "component":component, "hf_data": hf_data, "lf_data": lf_data, "x_pred":x_pred,
                          "dimension": dimension, "lengthscale":lengthscale, "fix_lengthscale":fix_lengthscale,
                          "fix_hf_noise":fix_hf_noise, "verbose":verbose }
        
        lambdas_pred, hf_dh_dl, hf_var_dh_dl = get_hf_prediction_mf( **settings_dict )

        # Overwrite the class wide composition, lambda, derivative, and variances of derivative lists
        self.compositions = x_pred if dimension == 2 else np.unique( self.compositions )
        self.lambdas      = [ lambdas_pred for _ in range(len(self.compositions))]
        self.dh_dl        = [ hf_dh_dl[ii*len(lambdas_pred):(ii+1)*len(lambdas_pred),0] for ii in range(len(self.compositions))]
        self.var_dh_dl    = [ hf_var_dh_dl[ii*len(lambdas_pred):(ii+1)*len(lambdas_pred),0] for ii in range(len(self.compositions))]
        self.overwriten   = True

    def gpr(self, component: str, dimension: int=2, x_pred: np.ndarray=np.array([]), lengthscale: List[float]=[], 
            fix_lengthscale: bool=False, fix_hf_noise: bool=True, verbose: bool=False):
        """
        Function that uses GPR modeling to interpolate high fidelity data.

        Args:
            component (str): Component for which the multi fidelity modeling is performed.
            dimension (int, optional): Dimension of the multi fidelity modeling. 2D includes the compositions, 1D is just dH/dl over lambda. Defaults to 2.
            x_pred (np.ndarray): New evaluated liquid compositions (in case of 2D). Defaults to np.array([]).
            lengthscale (List[float], optional): Possible default lenghtscale hyperparameters. Defaults to [].
            fix_lengthscale (bool, optional): If the lenghtscale hyperparameters should be fixed. Defaults to False.
            fix_hf_noise (bool, optional): If the noise of the high fidelity should be fixed. Defaults to True.
            verbose (bool, optional): If detailed information should be printed out. Defaults to False.                                            
        """

        # Call the multifidelity class to obtain interpolated high fidelity results
        hf_data = [ self.lambdas, self.compositions, self.dh_dl, self.var_dh_dl ]
        
        settings_dict = { "component":component, "hf_data": hf_data, "x_pred":x_pred, "dimension": dimension, 
                          "lengthscale":lengthscale, "fix_lengthscale":fix_lengthscale, "fix_hf_noise":fix_hf_noise, "verbose":verbose }
        
        lambdas_pred, hf_dh_dl, hf_var_dh_dl = get_hf_prediction_gpr( **settings_dict )

        # Overwrite the class wide composition, lambda, derivative, and variances of derivative lists
        self.compositions = x_pred if dimension == 2 else np.unique( self.compositions )
        self.lambdas      = [ lambdas_pred for _ in range(len(self.compositions))]
        self.dh_dl        = [ hf_dh_dl[ii*len(lambdas_pred):(ii+1)*len(lambdas_pred),0] for ii in range(len(self.compositions))]
        self.var_dh_dl    = [ hf_var_dh_dl[ii*len(lambdas_pred):(ii+1)*len(lambdas_pred),0] for ii in range(len(self.compositions))]
        self.overwriten   = True

    def plot(self,labels=["","$\lambda$","$ \\langle \\frac{\partial U}{\partial \lambda} \\rangle_{\lambda} \ / \ (k_\mathrm{B}T)$"], save_path: str="",
             plot_3d: bool=False, label_size=20, legend_size=16, tick_size=15, size=(12,10), linewidth=3, markersize=12 ):
        """
        Function that plots each dh_dl plot or one 3d plot

        Args:
            labels (list, optional): Data and ax label. Defaults to ["","$\lambda$","$ \langle \frac{\partial U}{\partial \lambda} \rangle_{\lambda} \ / \ (k_\mathrm{B}T)$"].
            save_path (str, optional): Saving destination. Plots will the saved there as "dh_dl_x*.png if provided. Defaults to "".
            plot_3d (bool, optional): If a 3d plot should be made for lambda, composition and dh_dl. Defaults to False.
        """
        if plot_3d: ax = plt.figure(figsize=size).add_subplot(projection='3d')

        for xi, lambdas, dh_dl, var_dh_dl in zip( np.unique( self.compositions ), self.lambdas, self.dh_dl, self.var_dh_dl ):
            
            # In case that the original sampling data is overwritten, add the original sampling data in the plot
            if self.overwriten:
                idx = np.where( np.unique( self.sampled_compositions ) == xi )
                if len(idx) > 0:
                    idx = idx[0]
                    lambdas_orig   = self.sampled_lambdas[idx]
                    dh_dl_orig     = self.sampled_dh_dl[idx]
                    var_dh_dl_orig = self.sampled_var_dh_dl[idx]

            if plot_3d:
                if not self.overwriten: 
                    ax.plot( lambdas, np.ones(len(lambdas))*xi, dh_dl, linestyle = "-", color="tab:blue", linewidth=linewidth, marker = ".", markersize=markersize)
                else:
                    ax.plot( lambdas, np.ones(len(lambdas))*xi, dh_dl, linestyle = "-", color="tab:blue", linewidth=linewidth)
                    ax.plot( lambdas_orig, np.ones(len(lambdas_orig))*xi, dh_dl_orig, linestyle = "None", marker = ".", color="tab:blue", markersize=markersize)
            else:
                print(f"Composition: {xi}\n")
                if not self.overwriten:
                    data   = [ [ lambdas, dh_dl, None, np.sqrt(var_dh_dl) ] ]
                    colors = [ "tab:blue" ]
                    marker = [ "." ]
                    line   = [ "-" ]
                    label  = labels
                else:
                    data   = [ [ lambdas, dh_dl, None, np.sqrt(var_dh_dl) ], [ lambdas_orig, dh_dl_orig, None, np.sqrt(var_dh_dl_orig) ] ]
                    colors = [ "tab:blue", "tab:blue" ]
                    marker = [ "None", "." ]
                    line   = [ "-", "None" ]
                    label  = [ "high fidelity", "simulation"] + labels[-2:]

                path_out = save_path + f"dh_dl_x{xi}.png" if save_path else ""
            
                plot_data( data, label, colors, path_out = path_out, linestyle = line, markerstyle = marker, ax_lim = [[0,1]])

        if plot_3d:
                
            # Make legend, set axes limits and labels
            if self.overwriten:
                ax.legend(["high fidelity", "simulation"],fontsize=legend_size,frameon=False)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            ax.dist = 12
            ax.w_xaxis.line.set_linewidth(2)
            ax.w_yaxis.line.set_linewidth(2)
            ax.w_zaxis.line.set_linewidth(2)

            ax.minorticks_on()
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.zaxis.set_minor_locator(AutoMinorLocator(2))

            ax.tick_params(which="major",labelsize=tick_size)
            ax.tick_params(which="minor",labelsize=tick_size)

            ax.set_xlabel(labels[-3],fontsize=label_size,labelpad=20)
            ax.set_ylabel(labels[-2],fontsize=label_size,labelpad=20)
            ax.set_zlabel(labels[-1],fontsize=label_size,labelpad=20)

            path_out = save_path + "dh_dl_all.png" if save_path else ""

            if path_out:
                plt.savefig(path_out,dpi=400)
            
            plt.show()
            plt.close()
        return
    
    def model_dump(self):
        """
        Function that dumps the class objects in a JSON serializable format
        """
        obj_dict = serialize_json( self.__dict__, target_class = () )
            
        return obj_dict

class FreeEnergyPerturbation:
    def __init__(self) -> None:
        pass


class MixtureComponent:
    def __init__(self, component: str, liquid_composition: List[float], temperature: List[float], mass_density: List[float], 
                 solvation_free_energy_contributions: Dict[ str, Dict[ str, List[float]]], molecular_mass: List[float] ) -> None:
        """
        Class that inherits all important information of a mixture component.

        Args:
            component (str): Name of the component.
            liquid_composition (List[float]): List of liquid mixture compositions (given in mol/mol).
            temperature (List[float]): Temperatures at the specified compositions (given in K).
            mass_density (List[float]): Mass densities at the specified compositions (given in kg/m^3). 
            solvation_free_energy_contributions (Dict[ str, Dict[ str, List[float]]]): Solvation free energy contributions and variances (van der Waals / Coulomb) of 
                                                                                       that component at the specified compositions (given in J/mol).
            molecular_mass (List[float]): Molecular mass of the mixture at the specified compositions (given in kg/mol).
        """
        
        # Define class wide attributes
        self.component                  = component
        self.liquid_composition         = np.array( liquid_composition )
        self.temperature                = np.array( temperature )
        self.mass_density               = np.array( mass_density )

        self.solvation_free_energy_contributions = solvation_free_energy_contributions
        
        # Get the total solvation free energy as sum over both contributions
        self.solvation_free_energy      = np.sum( [sfec[1]["delta_G"] for sfec in solvation_free_energy_contributions.items() ], axis = 0)
        self.solvation_free_energy_std  = np.sqrt( np.sum( [sfec[1]["var_delta_G"] for sfec in solvation_free_energy_contributions.items() ], axis = 0) )
        self.molecular_mass             = np.array( molecular_mass )

        # Assign pure substance properties
        idx                             = [ i for i,xi in enumerate(liquid_composition) if xi == 1.0 ][0]
        
        self.pure_temperature               = temperature[idx]
        self.pure_mass_density              = mass_density[idx]
        self.self_solvation_free_energy     = self.solvation_free_energy[idx]
        self.self_solvation_free_energy_std = self.solvation_free_energy_std[idx]
        self.pure_molecular_mass            = molecular_mass[idx]

        # Compute molar density
        self.pure_molar_density = self.pure_mass_density / self.pure_molecular_mass
        self.molar_density      = self.mass_density / self.molecular_mass

        # Define a free energy class dictionary to save the free energy objects for each contribution
        self.free_energy_object = {}

    def compute_gamma(self):
        """
        Function that computes the activity coefficient of the component in a mixture using solvation free energy:

        gamma_i = np.exp( ( delta G_i^{solv,i+j} - delta G_i^{solv,i} ) / (RT) ) * rho_{i+j} / rho_i

        std(gamma_i) = gamma_i * beta * std( delta G_i^{solv,i+j} - delta G_i^{solv,i} )

        Error propagation explained: gamma_i = exp( ( delta G_i^{solv,i+j} - delta G_i^{solv,i} ) * beta ) * rho_{i+j} / rho_i
                                     with std(exp(k*u)) / exp(k*u) = k*std(u) and std(k - u) = sqrt( std(k)**2 + std(u)**2 )
        """

        # Computation of gamma and its standard deviation
        self.gamma     = np.exp( ( self.solvation_free_energy - self.self_solvation_free_energy ) / ( R * self.temperature ) ) * self.molar_density / self.pure_molar_density
        self.gamma_std = self.gamma / ( R * self.temperature ) * np.sqrt( self.solvation_free_energy_std**2 + self.self_solvation_free_energy_std**2 )

        # Constrain standard deviation at pure substance to 0
        self.gamma_std[-1] = 0.0

    def compute_vapor_pressure(self):
        """
        Function that computes the pure vapor pressure of the component using solvation free energy.

        p_i = exp( delta G_i^{solv,i} / ( RT ) ) * RT * rho_i

        Error propagation explained:

        std(p_i) = p_i * std(delta G_i^{solv,i}) * beta
        with std(exp(k*u)) / exp(k*u) = k*std(u)

        """
        
        self.pure_vapor_pressure     = R * self.pure_temperature * self.pure_molar_density * np.exp( self.self_solvation_free_energy / ( R * self.pure_temperature ) )
        self.pure_vapor_pressure_std = self.pure_vapor_pressure * self.self_solvation_free_energy_std / ( R * self.pure_temperature ) 


    def add_reference_gamma(self, reference_composition: List[float], reference_gamma: List[float], reference_type: str="PC-SAFT" ):
        """Function that saves reference activiy coeffiecients in the class.

        Args:
            reference_composition (List[float]): Reference compositions.
            reference_gamma (List[float]): Reference coefficients.
            reference_type (str, optional): Origin of reference data.
        """
        self.gamma_composition_reference = reference_composition
        self.gamma_reference             = reference_gamma
        self.gamma_type_reference        = reference_type


    def add_free_energy_object(self, key: str, free_energy_object: ThermodynamicIntegration | FreeEnergyPerturbation ):
        """
        Function that adds the underlying free energy object to the class

        Args:
            key (str): String which free energy portion is investigated: vdw or coulomb
            free_energy_object (ThermodynamicIntegration | FreeEnergyPerturbation): Free energy class.
        """
        self.free_energy_object[key] = free_energy_object

    def add_attribute(self, key: str, attribute: float, attribute_std: float ):
        """
        Function that variable add class objects. These objects contain a value itself and a standard deviaton.

        Args:
            key (str): Attributes name in the class (allowed are: "seperation_factor", "vapor_compositions", "equilibrium_pressure" )
            attribute (float): Value of the attribute.
            attribute_std (float): Standard deviation of the attribute.
        """

        # Define allowed attributes
        attribute_list = [ "seperation_factor", "vapor_composition", "equilibrium_pressure" ]

        if key in attribute_list:
            setattr(self, key, attribute)
            setattr(self, f"{key}_std", attribute_std)
        else:
            raise KeyError("Specified attribute is not allowed! Allowed are: " + " ".join(attribute_list))
        
    def model_dump(self):
        """
        Function that dumps the class objects in a JSON serializable format
        """
        obj_dict = serialize_json( self.__dict__, ( ThermodynamicIntegration, FreeEnergyPerturbation ) )
            
        return obj_dict
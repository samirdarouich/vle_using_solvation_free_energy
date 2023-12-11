import numpy as np
from typing import List, Tuple
from scipy.constants import R
from scipy.interpolate import interp1d
from tools.utils import get_simulation_data, plot_data, trapezoid_integration, cubic_integration
from tools.multi_fidelity import get_lf_training_data, prep_mf_input, MF

def get_delta_fe(sim_path: str, components: List[str], charged: List[bool], free_eng_method: str,
                 compositions: List, no_composition: int, temperatures: List,
                 sim_lambdas: List[List], delta: float, both_ways: bool, dG_save_path: str,
                 lf_databanks: List[str], lf_mixtures: List[str], lf_unique_keys: List[str],
                 lengthscales: List[List]=[[],[]], fix_lengthscale: bool=False, fix_hf_noise: bool=True,
                 fraction: float= 0.0, verbose: bool=False ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Function that gather simulation data (dH/dlambda) to perform thermodynamic integration. 

    Args:
        sim_path (str): Path to simulation, should look like this: ".../%s/sim_%s/x%.1f/TI/sim_%s_%d/fep_%s.fep%d%d".
        components (List[str]): Names of the components of the mixture.
        charged (List[bool]): List if the components are charged (only relevant in Coulomb case).
        free_eng_method (str): Method for evaluation of the the free energy difference.
        compositions (List): Simulated compositions of this mixture.
        no_composition (int): Number of compositions for which the 2D multi fidelity should be evaluated
        temperatures (List): Simulated temperatures. Will be interpolated if more compositions then simulated are evaluated.
        sim_lambdas (List[List]): List containing sublists for the lambda intermediates used in the vdW/coulomb case.
        delta (float): Infitesimal small deflection to compute numerical derivative.
        both_ways (bool): If only a forward difference or a also a backward difference is performed.
        lf_databanks (List[str]): Path to low fidelity databank (for vdW / Coulomb ).
        lf_mixtures (List[str]): Mixture that should be utilized as low fidelity model (named: component1_component2) (for vdW / Coulomb ).
        lf_unique_keys (List[str]): Thermodynamic key for this mixture (e.g.: Temperature or pressure) (for vdW / Coulomb ).
        lengthscales (List[List], optional): Possible default lenghtscale hyperparameters (for vdW / Coulomb ). Defaults to [[],[]].
        fix_lengthscale (bool, optional): If the lenghtscale hyperparameters should be fixed. Defaults to False.
        fix_hf_noise (bool, optional): If the noise of the high fidelity should be fixed. Defaults to True.
        dG_save_path (str): Fillable path where plots should be saved.
        fraction (float, optional): Time fraction of simulation output that should be ommited. Defaults to 0.0.
        verbose (bool, optional): If detailed information should be printed out. Defaults to False.

    Returns:
        dG_mix (2D array): Free energy difference for each component in each composition.
        dG_var (2D array): Standard deviation of the free energy difference for each component in each composition.
        densities (1D array): Mass mixture density for each composition (interpolated to match x_pred).
        temperatures (1D array): Temperatures at which the free energy differences are computed (interpolated to match x_pred).
        x_pred (1D array): New evaluated liquid compositions
    """

    ## Initialize ##

    x_pred          = np.linspace(0.0, 1.0, no_composition) if no_composition != len(compositions) else compositions
    dG_mix          = np.zeros((2,no_composition))
    dG_var          = np.zeros((2,no_composition))
    densities       = []
    flag_plot_total = False

    interpol        = "quadratic" if len( compositions ) == 3 else "cubic"
    rt              = R * interp1d( compositions, temperatures, kind=interpol )( x_pred )

    for k,sim_lambda in enumerate( sim_lambdas ):

        sim_txt = "vdw" if k==0 else "coulomb"

        # If both components arent charged, skip the whole coulomb evaluation
        if k==1 and not any( charged ): continue

        print(f"\n{sim_txt} part of the solvation free energy\n")

        # Get simulation data for whole mixture
        settings_dict = { "sim_txt": sim_txt, "sim_path":sim_path, "components": components, "charged":charged,
                          "compositions": compositions, "lambdas": sim_lambda, "delta": delta,
                          "both_ways": both_ways, "fraction": fraction }
        
        hf_l, hf_xi, hf_du_dl, hf_du_dl_var, dens_tmp = get_simulation_data( **settings_dict )

        ## Perform integration on simulation data ##

        if "multi_fidelity" in free_eng_method:

            settings_dict = { "components": components, "hf_data": [hf_l,hf_xi,hf_du_dl,hf_du_dl_var], 
                              "free_eng_method": free_eng_method, "x_pred": x_pred, "lf_databank": lf_databanks[k], 
                              "lf_mixture": lf_mixtures[k], "lf_unique_key": lf_unique_keys[k], "lengthscale":lengthscales[k], 
                              "fix_lengthscale": fix_lengthscale, "fix_hf_noise": fix_hf_noise, "verbose": verbose }
            
            dG_mix_tmp, dG_std_tmp =  mf_modeling_dh_dl( **settings_dict )

        # Flip data for 2nd component to match that x2=1-x1
        dG_mix_tmp[1] = np.flip(dG_mix_tmp[1])
        dG_std_tmp[1] = np.flip(dG_std_tmp[1])
        dens_tmp[1]   = np.flip(dens_tmp[1])

        # Average the density over insertion/deletion from component 1 in mixture and from component 2 in mixture
        # If only one component is charged and the other not, then density arrays will have the initial zero entry for this component
        # To prevent the mean to be distorted, copy the gather densities from the charged component
        dummy = [dens for dens in dens_tmp if all(dens)]
        if len(dummy) == 1: dens_tmp = np.array( dummy*2 )

        # Multiply with temperature to get corect dimension and plot delta G over compositions
        dG_mix_tmp *= rt
        dG_std_tmp *= rt

        ## Plot the data ##

        data   = [[x_pred,dG_mix_tmp[0]/1000], [x_pred,dG_mix_tmp[1]/1000],
                  [x_pred,[ (dG_mix_tmp[0]-1.*dG_std_tmp[0])/1000, (dG_mix_tmp[0]+1.*dG_std_tmp[0])/1000] ],
                  [x_pred,[ (dG_mix_tmp[1]-1.*dG_std_tmp[1])/1000, (dG_mix_tmp[1]+1.*dG_std_tmp[1])/1000] ],
                  [x_pred[-1],dG_mix_tmp[0][-1]/1000,None,dG_std_tmp[0][-1]/1000], [x_pred[0],dG_mix_tmp[1][0]/1000,None,dG_std_tmp[1][0]/1000] ]
        
        labels = [ "$\Delta G_\mathrm{%s}^\mathrm{mix}$"%components[0], 
                   "$\Delta G_\mathrm{%s}^\mathrm{mix}$"%components[1],
                   "",
                   "",
                   "$\Delta G_\mathrm{%s}^\mathrm{pure}$"%components[0], 
                   "$\Delta G_\mathrm{%s}^\mathrm{pure}$"%components[1], 
                   "$x_\mathrm{%s}$ / -"%components[0], "$\Delta G / kJ/mol"]
        
        colors = [ "tab:blue", "tab:orange", "tab:blue", "tab:orange", "tab:blue", "tab:orange" ]
        ls     = [ "solid", "solid", "None", "None", "None", "None"]
        marker = [ "None","None", "None","None", "x", "x" ]
        
        fill   = [False, False, True, True, False, False]

        save_path = dG_save_path%sim_txt

        plot_data(data,labels,colors,save_path,ax_lim=[[0.0,1.0]],linestyle=ls,markerstyle=marker,lr=True,fill=fill)

        # Update overall delta G and densities
        dG_mix    += dG_mix_tmp
        dG_var    += dG_std_tmp**2
        densities.append( np.mean(dens_tmp,axis=0) )

        if k==1: flag_plot_total = True

    ## Plot total free solvation energy ##

    if flag_plot_total:
        data   = [ [x_pred,dG_mix[0]/1000], [x_pred,dG_mix[1]/1000],
                   [x_pred,[ (dG_mix[0]-1.*np.sqrt(dG_var)[0])/1000, (dG_mix[0]+1.*np.sqrt(dG_var)[0])/1000] ],
                   [x_pred,[ (dG_mix[1]-1.*np.sqrt(dG_var)[1])/1000, (dG_mix[1]+1.*np.sqrt(dG_var)[1])/1000] ],
                   [x_pred[-1],dG_mix[0][-1]/1000,None,np.sqrt(dG_var)[0][-1]/1000], [x_pred[0],dG_mix[1][0]/1000,None,np.sqrt(dG_var)[1][0]/1000] ]
        
        labels = [ "$\Delta G_\mathrm{%s}^\mathrm{mix}$"%components[0], 
                   "$\Delta G_\mathrm{%s}^\mathrm{mix}$"%components[1],
                   "",
                   "",
                   "$\Delta G_\mathrm{%s}^\mathrm{pure}$"%components[0], 
                   "$\Delta G_\mathrm{%s}^\mathrm{pure}$"%components[1], 
                   "$x_\mathrm{%s}$ / -"%components[0], "$\Delta G / kJ/mol"]
        
        colors = [ "tab:blue", "tab:orange", "tab:blue", "tab:orange", "tab:blue", "tab:orange" ]
        ls     = [ "solid", "solid", "None", "None", "None", "None"]
        marker = [ "None","None", "None","None", "x", "x" ]
        
        fill   = [False, False, True, True, False, False]
        
        save_path = dG_save_path%"total"

        plot_data(data,labels,colors,save_path,ax_lim=[[0.0,1.0]],linestyle=ls,markerstyle=marker,lr=True,fill=fill)
    
    # Interpolate sampled densities to match new composition prediction
    densities    = np.mean( densities, axis=0 )
    densities    = interp1d( compositions, densities, kind=interpol )( x_pred )
    temperatures = rt/R

    return dG_mix,np.sqrt(dG_var),densities,temperatures,x_pred


def mf_modeling_dh_dl(components: List[str], hf_data: List[List], free_eng_method: str, x_pred: np.ndarray,
                      lf_databank: str, lf_mixture: str, lf_unique_key: str, lengthscale: List[float]=[],
                      fix_lengthscale: bool=False, fix_hf_noise: bool=True, verbose=False ) -> Tuple[ np.ndarray,np.ndarray ]:

    """
    Function that performs multi-fidelity modeling (either 1D or 2D) on thermodynamic integration data to compute free energy differences

    Args:
        components (List[str]): Names of the components of the mixture.
        hf_data (list): List containing several sublists. In the first sublist, the lambda training points are provided for every composition of component.
                        In the 2nd sublist, the composition training points are provided for every composition of component.
                        In the 3th sublist, the du_dlambda training data is provided for every composition of component.
                        (Optional) In the 4th sublist, the variance of the du_dlambda training data is provided for every composition of component.
        free_eng_method (str): Method for evaluation of the the free energy difference.
        x_pred (np.ndarray): New evaluated liquid compositions.
        lf_databank (str): Path to low fidelity databank.
        lf_mixture (str): Mixture that should be utilized as low fidelity model (named: component1_component2).
        lf_unique_key (str): Thermodynamic key for this mixture (e.g.: Temperature or pressure).
        lengthscale (List[float], optional): Possible default lenghtscale hyperparameters. Defaults to [].
        fix_lengthscale (bool, optional): If the lenghtscale hyperparameters should be fixed. Defaults to False.
        fix_hf_noise (bool, optional): If the noise of the high fidelity should be fixed. Defaults to True.
        verbose (bool, optional): If detailed information should be printed out. Defaults to False.

    Returns:
        dG_mix_tmp (2D array): Solvation free energy difference for each component in each composition.
        dG_std_tmp (2D array): Standard deviation of the solvation free energy difference for each component in each composition.
    """

    dG_mix_tmp = np.zeros((2,len(x_pred)))
    dG_std_tmp = np.zeros((2,len(x_pred)))
    
    # Perform integration on 2D or 1D MF simulation data (first dimension: lambda, second dimension: composition)
    dimension = 2 if free_eng_method == "2d_multi_fidelity" else 1

    # Get low fidelity data and prepare mf data input
    settings_dict = {"lf_databank": lf_databank, "lf_mixture": lf_mixture, "lf_unique_key":lf_unique_key }
    lf_data       = get_lf_training_data( **settings_dict )

    ## Get high fidelity prediction and integrate it ##

    for i in range(2):
        
        # Skip if high fidelity data is emtpy, for example in the case of coulomb interaction for uncharged component
        if not any(len(sublist) > 0 for sublist in hf_data[0][i]): continue

        settings_dict = { "component":components[i], "hf_data": [hf[i] for hf in hf_data], "lf_data": [lf[i] for lf in lf_data], "x_pred":x_pred,
                          "dimension": dimension, "lengthscale":lengthscale, "fix_lengthscale":fix_lengthscale,
                          "fix_hf_noise":fix_hf_noise, "verbose":verbose }
        l_pred,hf_mean,hf_var = get_hf_prediction( **settings_dict )
        
        for ii,xi in enumerate(x_pred):
            dG_mix_tmp[i][ii], dG_std_tmp[i][ii] = cubic_integration( l_pred, hf_mean[ii*len(l_pred):(ii+1)*len(l_pred),0], hf_var[ii*len(l_pred):(ii+1)*len(l_pred),0] )

    return dG_mix_tmp, dG_std_tmp


def get_hf_prediction( component, hf_data: List[List], lf_data: List[List], x_pred: np.ndarray, dimension: int=2, 
                       lengthscale: List[float]=[], fix_lengthscale: bool=False, fix_hf_noise: bool=True, 
                       plot_3d: bool=True, no_lambda_intermediates: int=51, verbose=False) -> Tuple[ np.ndarray, np.ndarray, np.ndarray ]:
    """
    Function that uses linear Multi-fidelity modeling to interpolate dH/dlambda from free energy simulations. 
    This will loop through every composition of the mixture and return the interpolated dH/dlambda as well as its variance for the specified component.


    Args:
        component (_type_): Component for which the multi fidelity modeling is performed
        hf_data (list): List containing several sublists. In the first sublist, the lambda training points are provided for every composition of component.
                        In the 2nd sublist, the composition training points are provided for every composition of component.
                        In the 3th sublist, the du_dlambda training data is provided for every composition of component.
                        (Optional) In the 4th sublist, the variance of the du_dlambda training data is provided for every composition of component.
        lf_data (list): List containing several sublists. In the first sublist, the low fidelity lambdas are provided for every composition of low fidelity component.
                        In the 2nd sublist, the low fidelity compositions are provided for every composition of low fidelity component.
                        In the 3th sublist, the du_dlambda low fidelity data is provided for every composition of low fidelity component.
        x_pred (np.ndarray): New evaluated liquid compositions.
        dimension (int, optional): Dimension of the multi fidelity modeling. 2D includes the compositions, 1D is just dH/dl over lambda. Defaults to 2.
        lengthscale (List[float], optional): Possible default lenghtscale hyperparameters. Defaults to [].
        fix_lengthscale (bool, optional): If the lenghtscale hyperparameters should be fixed. Defaults to False.
        fix_hf_noise (bool, optional): If the noise of the high fidelity should be fixed. Defaults to True.
        plot_3d (bool, optional): If a 3D plot for the 2D modeling should be made or one plot for each composition. Defaults to True.
        no_lambda_intermediates (int, optional): New evaluated lambda intermediates. Defaults to 51.
        verbose (bool, optional): If detailed information should be printed out. Defaults to False.

    Returns:
        l_pred (1D array): Predicted lambda points
        hf_mean (2D array): Predicted du_dlambda data
        hf_var (2D array): Variance of predicted du_dlambda data
    """

    # Create lambda vector for multi fidelity prediction
    l_pred = np.linspace(0.0, 1.0, no_lambda_intermediates)

    if dimension == 2:

        print(f"\n2D Multi-fidelity modeling for component: {component}\n")

        # Prepare multi fidelity data as 2D data
        X_train,Y_train,F_train = prep_mf_input( hf_data[:3], lf_data, dimension=dimension )

        # Train multi fidelity model 
        mf_modeling             = MF( X_train, Y_train, F_train, n_optimization_restarts = 2 )

        # Set initial values (might results in faster and better convergence of hyperparameter optimization)
        if bool(lengthscale):
            mf_modeling.gpy_lin_mf_model.multifidelity.Mat32.lengthscale   = lengthscale
            mf_modeling.gpy_lin_mf_model.multifidelity.Mat32_1.lengthscale = lengthscale

        # If wanted constrain hyperparameters 
        if fix_lengthscale:
            mf_modeling.gpy_lin_mf_model.multifidelity.Mat32.lengthscale.fix()
            mf_modeling.gpy_lin_mf_model.multifidelity.Mat32_1.lengthscale.fix()
        
        # Use high fidelity simulation variance to fix noise of high fidelity
        if len(hf_data) == 4 and fix_hf_noise:
            mf_modeling.gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.variance = np.mean( hf_data[3] ) / mf_modeling.Y_normer**2 
            mf_modeling.gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.variance.fix()

        # Optimize hyperparemeters
        mf_modeling.train()

        if verbose:
            print("\nLow fidelity variance: %.3f"%mf_modeling.gpy_lin_mf_model[0])
            print("Low fidelity lengthscales: %.3f, %.3f"%(mf_modeling.gpy_lin_mf_model[1],mf_modeling.gpy_lin_mf_model[2]))
            print("High fidelity variance: %.3f"%mf_modeling.gpy_lin_mf_model[3])
            print("High fidelity lengthscales: %.3f, %.3f"%(mf_modeling.gpy_lin_mf_model[4],mf_modeling.gpy_lin_mf_model[5]))
            print("Phi parameter: %.3f"%mf_modeling.gpy_lin_mf_model[6])
            print("Low fidelity noise: %.3f"%mf_modeling.gpy_lin_mf_model[7])
            print("High fidelity noise: %.3f"%mf_modeling.gpy_lin_mf_model[8])
        
        ## Acquire high fidelity prediction ##

        col1, col2     = np.meshgrid(l_pred, x_pred)
        X_pred         = np.column_stack((col1.ravel(), col2.ravel()))

        hf_mean,hf_var = mf_modeling.predict_high_fidelity(X_pred)

        if verbose: 
            labels = ["$\lambda$ / -", "x$_\mathrm{%s}$ / -"%component,"$ \\langle \\frac{\partial U}{\partial \lambda} \\rangle_{\lambda} \ / \ (k_\mathrm{B}T)$"]
            mf_modeling.plot( labels, plot_3d = plot_3d )

    else:

        print(f"\n1D Multi-fidelity modeling for component: {component}\n")
        hf_mean,hf_var = [],[]

        # *hf_var_du_dl syntax is used to capture the remaining elements in a list. If none are left, then hf_var_du_dl will be empty
        for hf_l, hf_x, hf_du_dl, *hf_var_du_dl in zip( *hf_data ):

            # Seach for low fidelity data that fits composition the best
            idx = np.argmin( [np.abs(np.unique(hf_x) - np.unique(lf_x)) for lf_x in lf_data[1] ] )

            # Prepare multi fidelity data as 1D data
            X_train,Y_train,F_train = prep_mf_input( [hf_l, hf_du_dl], [lf_data[0][idx],lf_data[2][idx]], dimension=dimension )

            # Train multi fidelity model
            mf_modeling = MF( X_train, Y_train, F_train, n_optimization_restarts = 2 )

            # Set initial values (might results in faster and better convergence of hyperparameter optimization)
            if bool(lengthscale):
                mf_modeling.gpy_lin_mf_model.multifidelity.Mat32.lengthscale   = lengthscale[0]
                mf_modeling.gpy_lin_mf_model.multifidelity.Mat32_1.lengthscale = lengthscale[0]
            
            # If wanted constrain hyperparameters 
            if fix_lengthscale:
                mf_modeling.gpy_lin_mf_model.multifidelity.Mat32.lengthscale.fix()
                mf_modeling.gpy_lin_mf_model.multifidelity.Mat32_1.lengthscale.fix()
            
            # Use high fidelity simulation variance to fix noise of high fidelity #
            
            if len(hf_data) == 4 and fix_hf_noise:
                mf_modeling.gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.variance = np.mean( hf_var_du_dl ) / mf_modeling.Y_normer**2 
                mf_modeling.gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.variance.fix()

            # Optimize hyperparemeters #
            mf_modeling.train()

            if verbose:
                print("\nLow fidelity variance: %.3f"%mf_modeling.gpy_lin_mf_model[0])
                print("Low fidelity lengthscale: %.3f"%mf_modeling.gpy_lin_mf_model[1])
                print("High fidelity variance: %.3f"%mf_modeling.gpy_lin_mf_model[2])
                print("High fidelity lengthscale: %.3f"%mf_modeling.gpy_lin_mf_model[3])
                print("Phi parameter: %.3f"%mf_modeling.gpy_lin_mf_model[4])
                print("Low fidelity noise: %.3f"%mf_modeling.gpy_lin_mf_model[5])
                print("High fidelity noise: %.3f"%mf_modeling.gpy_lin_mf_model[6])

            ## Acquire high fidelity prediction ##

            X_pred    = l_pred.reshape(-1,1)
            hf_m,hf_v = mf_modeling.predict_high_fidelity(X_pred)

            hf_mean.append(hf_m)
            hf_var.append(hf_v)

            if verbose: 
                labels = ["$\lambda$ / -", "$ \\langle \\frac{\partial U}{\partial \lambda} \\rangle_{\lambda} \ / \ (k_\mathrm{B}T)$"]
                mf_modeling.plot(labels)

        # Concatenate all high fidelity predictions for every high fidelity composition
        hf_mean, hf_var = np.concatenate(hf_mean), np.concatenate(hf_var)
        
    return l_pred, hf_mean, hf_var

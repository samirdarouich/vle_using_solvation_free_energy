import numpy as np
from scipy.interpolate import interp1d
from tools.utils import get_simulation_data, plot_data, trapezoid_integration, cubic_integration
from tools.multi_fidelity import get_lf_training_data, prep_mf_input, MF

def get_delta_fe(settings,print_out=False):
    """
    Function that gather simulation data (dU/dlambda) to perform thermodynamic integration. 

    Args:
        settings (dict): Dictionary containing all the necessary input.
        print_out (bool, optional): If detailed information should be printed out. Defaults to False.

    Returns:
        dG_mix (2D array): Free energy difference for each component in each composition.
        dG_var (2D array): Standard deviation of the free energy difference for each component in each composition.
        densities (1D array): Mass mixture density for each composition (interpolated to match x_pred).
        temperatures (1D array): Temperatures at which the free energy differences are computed (interpolated to match x_pred).
        x_pred (1D array): New evaluated liquid compositions
    """

    ## Initialize ##

    n_mix           = settings["no_composition"]
    x_pred          = np.linspace(0.0, 1.0, n_mix) if n_mix != len(settings["compositions"]) else np.array(settings["compositions"])
    dG_mix          = np.zeros((2,n_mix))
    dG_var          = np.zeros((2,n_mix))
    densities       = []
    flag_plot_total = False

    interpol        = "quadratic" if len( settings["compositions"] ) == 3 else "cubic"
    rt              = 8.314*interp1d( settings["compositions"], settings["temperatures"], kind=interpol )( x_pred )

    for k,sim_lambda in enumerate( settings["sim_lambdas"] ):

        settings["lambda"]        = sim_lambda
        settings["intermediates"] = len(sim_lambda)
        settings["lf_key"]        = "vdW" if k==0 else "coulomb"

        # If both components arent charged, skip the whole evaluation
        if k==1 and not any( settings["charged"] ): continue

        print("\n%s part of the free solvation energy\n"%settings["lf_key"])

        ## Get simulation data for whole mixture ##

        hf_l,hf_xi,hf_du_dl,hf_du_dl_var,dens_tmp = get_simulation_data( settings["sim_txts"][k],settings["sim_txt_files"][k], settings )

        ## Perform integration on simulation data ##

        if "multi_fidelity" in settings["free_eng_method"]:
            dG_mix_tmp,dG_std_tmp =  mf_modeling([hf_l,hf_xi,hf_du_dl,hf_du_dl_var],settings,n_mix,k,print_out=print_out)

        ## Flip data for 2nd component to match that x2=1-x1 ##

        dG_mix_tmp[1] = np.flip(dG_mix_tmp[1])
        dG_std_tmp[1] = np.flip(dG_std_tmp[1])
        dens_tmp[1]   = np.flip(dens_tmp[1])

        ## Average the density over insertion/deletion from component 1 in mixture and from component 2 in mixture ##
        # If only one component is charged and the other not, then density arrays will have the initial zero entry for this component #
        # To prevent the mean to be distorted, copy the gather densities from the charged component #
        dummy = [dens for dens in dens_tmp if all(dens)]
        if len(dummy) == 1: dens_tmp = np.array( dummy*2 )

        ## Multiply with temperature to get corect dimension and plot delta G over compositions ##

        dG_mix_tmp *= rt
        dG_std_tmp *= rt

        data   = [[x_pred,dG_mix_tmp[0]/1000], [x_pred,dG_mix_tmp[1]/1000],
                  [x_pred,[ (dG_mix_tmp[0]-1.*dG_std_tmp[0])/1000, (dG_mix_tmp[0]+1.*dG_std_tmp[0])/1000] ],
                  [x_pred,[ (dG_mix_tmp[1]-1.*dG_std_tmp[1])/1000, (dG_mix_tmp[1]+1.*dG_std_tmp[1])/1000] ],
                  [x_pred[-1],dG_mix_tmp[0][-1]/1000,None,dG_std_tmp[0][-1]/1000], [x_pred[0],dG_mix_tmp[1][0]/1000,None,dG_std_tmp[1][0]/1000] ]
        
        labels = [ "$\Delta G_\mathrm{%s}^\mathrm{mix}$"%settings["components"][0], 
                   "$\Delta G_\mathrm{%s}^\mathrm{mix}$"%settings["components"][1],
                   "",
                   "",
                   "$\Delta G_\mathrm{%s}^\mathrm{pure}$"%settings["components"][0], 
                   "$\Delta G_\mathrm{%s}^\mathrm{pure}$"%settings["components"][1], 
                   "$x_\mathrm{%s}$ / -"%settings["components"][0], "$\Delta G$ / kJ/mol" ]
        
        colors = [ "tab:blue", "tab:orange", "tab:blue", "tab:orange", "tab:blue", "tab:orange" ]
        ls     = [ "solid", "solid", "None", "None", "None", "None"]
        marker = [ "None","None", "None","None", "x", "x" ]
        
        fill   = [False, False, True, True, False, False]

        save_path = settings["dG_save_path"]%settings["sim_plts"][k]

        plot_data(data,labels,colors,save_path,ax_lim=[[0.0,1.0]],linestyle=ls,markerstyle=marker,lr=True,fill=fill)

        ## Update overall delta G and densities ##

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
        
        labels = [ "$\Delta G_\mathrm{%s}^\mathrm{mix}$"%settings["components"][0], 
                   "$\Delta G_\mathrm{%s}^\mathrm{mix}$"%settings["components"][1],
                   "",
                   "",
                   "$\Delta G_\mathrm{%s}^\mathrm{pure}$"%settings["components"][0], 
                   "$\Delta G_\mathrm{%s}^\mathrm{pure}$"%settings["components"][1], 
                   "$x_\mathrm{%s}$ / -"%settings["components"][0], "$\Delta G$ / kJ/mol" ]
        
        colors = [ "tab:blue", "tab:orange", "tab:blue", "tab:orange", "tab:blue", "tab:orange" ]
        ls     = [ "solid", "solid", "None", "None", "None", "None"]
        marker = [ "None","None", "None","None", "x", "x" ]
        
        fill   = [False, False, True, True, False, False]
        
        save_path = settings["dG_save_path"]%"total"

        plot_data(data,labels,colors,save_path,ax_lim=[[0.0,1.0]],linestyle=ls,markerstyle=marker,lr=True,fill=fill)
    

    ## Interpolate sampled densities to match new composition prediction ##
    
    densities    = np.mean( densities, axis=0 )
    densities    = interp1d( settings["compositions"], densities, kind=interpol )( x_pred )
    temperatures = rt/8.314

    return dG_mix,np.sqrt(dG_var),densities,temperatures,x_pred


def mf_modeling(hf_data,settings,n_mix,k,print_out=False):
    """
    Function that performs multi-fidelity modeling (either 1D or 2D) on thermodynamic integration data to compute free energy differences

    Args:
        hf_data (list): List containing the high fidelity data. 1D: (lambdas,du_dlambda); 2D: (lambdas,composition,du_dlambda)
        settings (dict): Dictionary containing all necessary information
        n_mix (int): Number of compositions for which the postprocessing should be performed
        print_out (bool, optional): If detailed information should be printed out. Defaults to False.

    Returns:
        dG_mix_tmp (2D array): Free solvation energy difference for each component in each composition.
        dG_std_tmp (2D array): Standard deviation of the free solvation energy difference for each component in each composition.
    """

    dG_mix_tmp = np.zeros((2,n_mix))
    dG_std_tmp = np.zeros((2,n_mix))
    

    # Perform integration on 2D or 1D MF simulation data (first dimension: lambda, second dimension: composition) #
    dimension = 2 if settings["free_eng_method"] == "2d_multi_fidelity" else 1

    ## Get high fidelity prediction and integrate it ##

    for i in range(2):

        if not settings["charged"][i] and k==1: continue

        l_pred,x_pred,hf_mean,hf_var = get_hf_prediction( hf_data, settings, k, i, dimension=dimension, n2=n_mix, print_out=print_out )
        
        for ii,xi in enumerate(x_pred):
            dG_mix_tmp[i][ii], dG_std_tmp[i][ii] = cubic_integration(l_pred, hf_mean[ii*len(l_pred):(ii+1)*len(l_pred),0], hf_var[ii*len(l_pred):(ii+1)*len(l_pred),0] )

    return dG_mix_tmp, dG_std_tmp


def get_hf_prediction(hf_data,settings,k,i,dimension=2,n1=51,n2=21,print_out=False):
    """
    Function that uses linear Multi-fidelity modeling to interpolate dH/dlambda from free energy simulations. 
    This will loop through every composition of the mixture and return the interpolated derivative as well as its variance for the specified component i.

    Args:
        hf_data (list): List containing several sublists. In the first sublist, the lambda training points are provided for every composition of component i.
                        In the 2nd sublist, the composition training points are provided for every composition of component i.
                        In the 3th sublist, the du_dlambda training data is provided for every composition of component i.
                        (Optional) In the 4th sublist, the variance of the du_dlambda training data is provided for every composition of component i.
        settings (dict): Dictionary containing all information needed for subfunctions
        k (int): Indicater for which portion of free energy (VdW or Coulomb --> just relevant if lengthscale should be constrained)
        i (int): Indicater for which component the high fidelity data should be provided
        dimension (int, optional): If a 2D (lambdas and composition) or 1D (lambdas) Multi-fidelity model should be trained. Defaults to 2.
        n1 (int, optional): Number of evaluated lambda points for high fidelity prediction. Defaults to 51.
        n2 (int, optional): Number of evaluated composition points for high fidelity prediction (only used in 2D case). Defaults to 21.
        print_out (bool, optional): Boolean if detailed information should be displayed. Defaults to False.

    Returns:
        l_pred (1D array): Predicted lambda points
        x_pred (1D array): Predicted composition points (only used in 2D MF case)
        hf_mean (2D array): Predicted du_dlambda data
        hf_var (2D array): Variance of predicted du_dlambda data
    """

    l_pred          = np.linspace(0.0, 1.0, n1)
    x_pred          = np.linspace(0.0, 1.0, n2)

    ## Get low fidelity data and prepare mf data input ##

    lf_l,lf_xi,lf_du_dl = get_lf_training_data( settings )

    if dimension == 2:

        print("\n2D Multi-fidelity modeling for component: %s\n"%settings["components"][i])
        # Prepare multi fidelity data as 2D data
        X_train,Y_train,F_train = prep_mf_input( [hf_data[0][i], hf_data[1][i], hf_data[2][i]], [lf_l[i], lf_xi[i], lf_du_dl[i]], dimension=dimension )

        ## Train multi fidelity model ##
        mf_modeling = MF(X_train,Y_train,F_train,n_optimization_restarts=2)

        # Constrain hyperparameter lenghtscale to be within 0 and 1 #
        
        #mf_modeling.gpy_lin_mf_model.multifidelity.Mat32.lengthscale.constrain_bounded(0.07,1.1,warning=False)
        #mf_modeling.gpy_lin_mf_model.multifidelity.Mat32_1.lengthscale.constrain_bounded(0.07,1.1,warning=False)

        if bool(settings["mf_length"][k]):
            mf_modeling.gpy_lin_mf_model.multifidelity.Mat32.lengthscale   = settings["mf_length"][k]
            mf_modeling.gpy_lin_mf_model.multifidelity.Mat32_1.lengthscale = settings["mf_length"][k]

        if settings["fix_length"]:
            mf_modeling.gpy_lin_mf_model.multifidelity.Mat32.lengthscale.fix()
            mf_modeling.gpy_lin_mf_model.multifidelity.Mat32_1.lengthscale.fix()
        
        # Use high fidelity simulation variance to fix noise of high fidelity #
        
        if len(hf_data) == 4 and settings["fix_hf_noise"]:
            mf_modeling.gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.variance = np.mean( hf_data[3][i] ) ##! this needs to be normed --> here we dont norm
            mf_modeling.gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.variance.fix()

        # Optimize hyperparemeters #
        mf_modeling.train()

        if print_out:
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

        if print_out: 
            labels = ["$\lambda$ / -", "x$_\mathrm{%s}$ / -"%settings["components"][0],"$ \\langle \\frac{\partial U}{\partial \lambda} \\rangle_{\lambda} \ / \ (k_\mathrm{B}T)$"]
            mf_modeling.plot(labels,plot_3d=settings["plot_3d"])

    else:

        print("\n1D Multi-fidelity modeling for component: %s\n"%settings["components"][i])
        hf_mean,hf_var = [],[]

        for hf_l,hf_x,hf_du_dl,hf_var_du_dl in zip( hf_data[0][i], hf_data[1][i], hf_data[2][i], hf_data[3][i] ):

            # Seach for low fidelity data that fits composition the best
            idx = np.argmin( [np.abs(np.unique(hf_x) - np.unique(lf_x)) for lf_x in lf_xi[i] ] )

            # Prepare multi fidelity data as 1D data
            X_train,Y_train,F_train = prep_mf_input( [hf_l, hf_du_dl], [lf_l[i][idx],lf_du_dl[i][idx]], dimension=dimension )

            ## Train multi fidelity model ##
            mf_modeling = MF(X_train,Y_train,F_train,n_optimization_restarts=2)

            if bool(settings["mf_length"][k]):
                mf_modeling.gpy_lin_mf_model.multifidelity.Mat32.lengthscale   = settings["mf_length"][k][0]
                mf_modeling.gpy_lin_mf_model.multifidelity.Mat32_1.lengthscale = settings["mf_length"][k][0]
                
            if settings["fix_length"]:
                mf_modeling.gpy_lin_mf_model.multifidelity.Mat32.lengthscale.fix()
                mf_modeling.gpy_lin_mf_model.multifidelity.Mat32_1.lengthscale.fix()
            
            # Use high fidelity simulation variance to fix noise of high fidelity #
            
            if len(hf_data) == 4 and settings["fix_hf_noise"]:
                mf_modeling.gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.variance = np.mean( hf_var_du_dl )
                mf_modeling.gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.variance.fix()

            # Optimize hyperparemeters #
            mf_modeling.train()

            if print_out:
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

            if print_out: 
                labels = ["$\lambda$ / -", "$ \\langle \\frac{\partial U}{\partial \lambda} \\rangle_{\lambda} \ / \ (k_\mathrm{B}T)$"]
                mf_modeling.plot(labels)

        # Concatenate all high fidelity predictions for every high fidelity composition
        hf_mean,hf_var = np.concatenate(hf_mean),np.concatenate(hf_var)
        
    return l_pred,x_pred,hf_mean,hf_var

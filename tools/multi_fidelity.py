import GPy
import emukit
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from matplotlib.ticker import AutoMinorLocator
from .general_utils import plot_data, work_json
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array


class MF():
    
    def __init__(self,X,Y,fidelity,mirror=False,kernel=GPy.kern.Matern32,n_optimization_restarts=1):
        """
        Class that initialize a Multi-fidelity approach

        Args:
            X (2D array): Training locations, can be multi dimensional (by appending more columns )
            Y (2D array): Training points, can be multi dimensional (by appending more columns )
            fidelity (2D array): Fidelity of training locations/pints
            mirror (boolean,optional): If training points are symmetric and should be mirrored. Defaults to False
            kernel (function, optional): Kernel function that should be used. Defaults to GPy.kern.Matern32.
            n_optimization_restarts (int, optional): Number of optimization restarts. Defaults to 1.
        """

        axis,axis_inv = 0,-1

        ## Prepare input data ##

        # Safe input data in class #
        self.X        = X
        self.Y        = Y
        self.fidelity = fidelity
        
        # Normalize data (frobenius norm for each input dimension) #
        self.X_normer = np.ones((1,X.shape[axis_inv])) #np.linalg.norm(self.X,axis=axis)
        self.Y_normer = np.ones((1,1)) #np.linalg.norm(self.Y,axis=axis)       
        
        self.X_norm   = self.X/self.X_normer
        self.Y_norm   = self.Y/self.Y_normer
        
        self.X_min    = np.min(self.X,axis=axis)
        self.X_max    = np.max(self.X,axis=axis)
        
        ## Mirror data to account for boundary conditions (optional) ## --> has to be checked if correct
        
        if mirror:
            # Make data continous
            
            Xd             = self.X_max - self.X_min

            self.X_train   = np.concatenate(( np.flip(self.X_norm-Xd,axis=axis), 
                                            self.X_norm, 
                                            np.flip(self.X_norm-Xd,axis=axis)), axis=axis)

            self.Y_train   = np.concatenate( [np.flip(self.Y_norm,axis=axis),
                                            self.Y_norm,
                                            np.flip(self.Y_norm,axis=axis)],axis=axis )
            
            fidelity_train = np.concatenate(( np.flip(self.fidelity,axis=axis), 
                                            self.fidelity, 
                                            np.flip(self.fidelity,axis=axis)), axis=axis)     
            
        else:
            self.X_train   = self.X_norm
            self.Y_train   = self.Y_norm
            fidelity_train = self.fidelity

        self.X_train_F     = np.concatenate( [ self.X_train, fidelity_train ], axis=axis_inv )

        ## Setup Multi-fidelity model ##
        
        self.dims              = self.X.shape[axis_inv]

        self.unique_fidelities = np.sort(np.unique(fidelity))
        self.no_fidelities     = len(self.unique_fidelities)

        kernels                = [kernel(self.dims) if self.dims==1 else kernel(self.dims,ARD=True) for _ in range(self.no_fidelities)]

        lin_mf_kernel          = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
        self.gpy_lin_mf_model  = GPyLinearMultiFidelityModel(self.X_train_F, self.Y_train, lin_mf_kernel, n_fidelities=self.no_fidelities)       
        self.lin_mf_model      = GPyMultiOutputWrapper(self.gpy_lin_mf_model, self.no_fidelities, n_optimization_restarts=n_optimization_restarts)

        return
        
    def train(self):
        """
        Train the multi fidelity object.
        
        Adjust and fix hyperparameters between initialisation and training of the model
        """
        self.lin_mf_model.optimize()    

        return

    def predict_high_fidelity(self, x ):
        """
        Predicts un-normed high fidelity results for a trained multi fidelity model
        
        x (2D array): evaluation locations, can be multi dimensional (by appending more columns )
            
        returns mean predictions and corresponding variances as np.arrays
        """        
        X  = x/self.X_normer
        X  = np.concatenate( [ X, np.ones((len(x),1)) * (self.no_fidelities-1) ], axis=-1 )

        hf_mean, hf_var = self.lin_mf_model.predict(X)

        return hf_mean*self.Y_normer, hf_var*self.Y_normer**2
    
    def predict_all_fidelities(self, x ):
        """
        predicts all un-normed results for a trained multi fidelity model
        
        x (2D array): evaluation locations, can be multi dimensional (by appending more columns )
            
        returns mean predictions and corresponding variances as np.arrays
        """        
        
        X  = x/self.X_normer  
        X  = convert_x_list_to_array([X]*self.no_fidelities)
        
        mean, var = self.lin_mf_model.predict(X)

        return mean*self.Y_normer, var*self.Y_normer**2
    
    def plot(self,labels,label_size=20,legend_size=16,tick_size=15,size=(12,10),colors=["tab:olive","tab:blue"],plot_3d=False,savepath=""):

        legend_label = ["low fidelity prediction","low fidelity data",
                        "high fidelity prediction","high fidelity data"]

        n,n1= 50,11

        msize=12
        lsize = 3     
        
        if self.dims == 2:
            if plot_3d: ax = plt.figure(figsize=size).add_subplot(projection='3d')

            for xi in np.linspace(self.X_min[1],self.X_max[1],n1):

                dummy = np.column_stack( ( np.linspace(self.X_min[0],self.X_max[0],n), np.ones(n)*xi ) )

                Y_predict, varY_predict = self.predict_all_fidelities(dummy)

                data = []

                for i in self.unique_fidelities:
                    mean = Y_predict[ int(i*n):int((i+1)*n),0 ]
                    var  = varY_predict[ int(i*n):int((i+1)*n),0 ]
                    ll   = dummy[:n,0]
                    xx   = dummy[:n,1]
                    p    = np.where(np.logical_and(self.X[:,1].reshape(-1,1) == np.round(xi,2), self.fidelity == i))

                    if plot_3d:
                        ax.plot(ll,xx, mean,"-",color=colors[int(i)],linewidth=lsize)
                        ax.plot(self.X[p], self.X[p,1][0],self.Y[p],".",markersize=msize,color=colors[int(i)] )
                    else:
                        # 95% confidence interval and mean prediction #
                        data.append([ ll, mean ] )
                        data.append([ ll, [ mean-1.96*np.sqrt(var), mean+1.96*np.sqrt(var) ] ] )
                        data.append([ self.X[p], self.Y[p] ] )

                if not plot_3d:
                    label_2d  = [ "Predicted low fidelity", "", "Training low fidelity",
                                "Predicted high fidelity", "", "Training high fidelity",
                                labels[0], labels[2]]
                    colors_2d = [ "tab:olive", "tab:olive", "tab:olive", "tab:blue", "tab:blue", "tab:blue"]
                    ls_2d     = [ "dashed", "None", "None", "dashed", "None", "None" ]
                    marker_2d = [ "None", "None", ".", "None", "None", "." ]
                    fill_2d   = [ False, True, False, False, True, False ]

                    plot_data(data,label_2d,colors_2d,savepath,ax_lim=[[0.0,1.0]],legend_size=14,linestyle=ls_2d,markerstyle=marker_2d,fill=fill_2d)
            
            if plot_3d:
                
                # Make legend, set axes limits and labels #
                ax.legend(legend_label,fontsize=legend_size,frameon=False)
                
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

                ax.set_xlabel(labels[0],fontsize=label_size,labelpad=20)
                ax.set_ylabel(labels[1],fontsize=label_size,labelpad=20)
                ax.set_zlabel(labels[2],fontsize=label_size,labelpad=20)

                if savepath: plt.savefig(savepath,dpi=400)
                plt.show()
                plt.close()
        else:

            dummy = np.linspace(self.X_min[0],self.X_max[0],n).reshape(-1,1)

            Y_predict, varY_predict = self.predict_all_fidelities(dummy)

            data = []

            for i in self.unique_fidelities:
                mean = Y_predict[ int(i*n):int((i+1)*n),0 ]
                var  = varY_predict[ int(i*n):int((i+1)*n),0 ]
                ll   = dummy[:n,0]
                p    = np.where(self.fidelity == i)
                data.append([ ll, mean ] )
                data.append([ ll, [ mean-1.96*np.sqrt(var), mean+1.96*np.sqrt(var) ] ] )
                data.append([ self.X[p], self.Y[p] ] )

            label_2d  = [ "Predicted low fidelity", "", "Training low fidelity",
                                "Predicted high fidelity", "", "Training high fidelity",
                                labels[0], labels[1]]
            colors_2d = [ "tab:olive", "tab:olive", "tab:olive", "tab:blue", "tab:blue", "tab:blue"]
            ls_2d     = [ "dashed", "None", "None", "dashed", "None", "None" ]
            marker_2d = [ "None", "None", ".", "None", "None", "." ]
            fill_2d   = [ False, True, False, False, True, False ]

            plot_data(data,label_2d,colors_2d,savepath,ax_lim=[[0.0,1.0]],legend_size=14,linestyle=ls_2d,markerstyle=marker_2d,fill=fill_2d)
         
        return
    

def get_lf_training_data(lf_databank: str, lf_mixture: str, lf_unique_key: str, lf_component: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Function that acquires low fidelity training data from given databank (json file). This is especially designed to read out the 
    derivative of the coupling work requiered to insert a molecule of component i into a mixture of i and j.

    Args:
        lf_databank (str): Path to low fidelity databank.
        lf_mixture (str): Mixture that should be utilized as low fidelity model.
        lf_unique_key (str): Thermodynamic key for this mixture (e.g.: Temperature or pressure).
        lf_component (str): Component which should be used as low fidelity model.

    Returns:
        lambda_learn (List[np.ndarray]): List containing lambda intermediates for every composition
        composition_learn (List[np.ndarray]): List containing composition intermediates for every composition
        dh_dl_learn (List[np.ndarray]): List containing derivative of the coupling work for every composition
    """

    lf_dict           = work_json(lf_databank,to_do="read")

    lambda_learn      = [ np.array(item) for item in lf_dict[lf_mixture][lf_unique_key][lf_component]["lambdas"] ]
    composition_learn = [ np.array(item) for item in lf_dict[lf_mixture][lf_unique_key][lf_component]["compositions"] ]
    dh_dl_learn       = [ np.array(item) for item in lf_dict[lf_mixture][lf_unique_key][lf_component]["dh_dl"] ]
    
    return lambda_learn, composition_learn, dh_dl_learn


def prep_mf_input(hf,lf,dimension):
    """
    Function that combines high fidelity and low fidelity data into the input needed for the Multi-fidelity Class.

    Args:
        hf (list): List containing as first entry a list with a 1D arrays with the learning lambdas of the high fidelity data,
                   2nd entry a list with 1D arrays with the learning compositions of the high fidelity data, 
                   3th entry a list with 1D arrays of learning values for the high fidelity data
        lf (list): List containing as first entry a list with a 1D arrays with the learning points of the low fidelity data, 
                   2nd entry a list with 1D arrays with the learning compositions of the low fidelity data, 
                   3th entry a list with 1D arrays of learning values for the low fidelity data
        dimension (int): Dimension of training points 

    Returns:
        X_train (2D array): Training points
        Y_train (2D array): Training values
        F_train (2D array): Fidelity of training points/valus
    """

    ## Prepare all fidelites together ##

    # Prepare input locations of low and high fidelity. If 2D, then combine lambdas and compositions, otherwise just use lambdas
    
    hf_x_comb = np.concatenate( [np.column_stack( ( _ ) ) for _ in zip( *hf[:dimension] ) ] )
    lf_x_comb = np.concatenate( [np.column_stack( ( _ ) ) for _ in zip( *lf[:dimension] ) ] )

    # Add low and high fidelity to one long list
    X_train = np.concatenate( (lf_x_comb, hf_x_comb ) )
    Y_train = np.concatenate( [ np.array(aa).reshape(-1,1) for aa in lf[-1] ]  + [ np.array(aa).reshape(-1,1) for aa in hf[-1] ] )
    F_train = np.zeros( (len(X_train), dimension-1 if dimension >= 2 else 1) )
    F_train[-len(hf_x_comb):] = 1.

    return X_train,Y_train,F_train

def get_hf_prediction( component: str, hf_data: List[List], lf_data: List[List], dimension: int=2, x_pred: np.ndarray=np.array([]),  
                       lengthscale: List[float]=[], fix_lengthscale: bool=False, fix_hf_noise: bool=True, 
                       plot_3d: bool=True, no_lambda_intermediates: int=51, verbose=False) -> Tuple[ np.ndarray, np.ndarray, np.ndarray ]:
    """
    Function that uses linear Multi-fidelity modeling to interpolate dH/dlambda from free energy simulations. 
    This will loop through every composition of the mixture and return the interpolated dH/dlambda as well as its variance for the specified component.


    Args:
        component (str): Component for which the multi fidelity modeling is performed.
        hf_data (list): List containing several sublists. In the first sublist, the lambda training points are provided for every composition of component.
                        In the 2nd sublist, the composition training points are provided for every composition of component.
                        In the 3th sublist, the du_dlambda training data is provided for every composition of component.
                        (Optional) In the 4th sublist, the variance of the du_dlambda training data is provided for every composition of component.
        lf_data (list): List containing several sublists. In the first sublist, the low fidelity lambdas are provided for every composition of low fidelity component.
                        In the 2nd sublist, the low fidelity compositions are provided for every composition of low fidelity component.
                        In the 3th sublist, the du_dlambda low fidelity data is provided for every composition of low fidelity component.
        dimension (int, optional): Dimension of the multi fidelity modeling. 2D includes the compositions, 1D is just dH/dl over lambda. Defaults to 2.
        x_pred (np.ndarray, optional): New evaluated liquid compositions (in case of 2D multi fidelity). Defaults to np.array([]).
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
        X_train,Y_train,F_train = prep_mf_input( hf_data[:3], lf_data[:3], dimension=dimension )

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
            
            # Use high fidelity simulation variance to fix noise of high fidelity 
            if len(hf_data) == 4 and fix_hf_noise:
                mf_modeling.gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.variance = np.mean( hf_var_du_dl ) / mf_modeling.Y_normer**2 
                mf_modeling.gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.variance.fix()

            # Optimize hyperparemeters
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

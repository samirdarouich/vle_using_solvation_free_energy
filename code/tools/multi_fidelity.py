import GPy
import numpy as np
import emukit
import matplotlib.pyplot as plt
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array
from matplotlib.ticker import AutoMinorLocator
from .utils import plot_data, work_json

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
    

def get_lf_training_data(settings):
    """
    Function that accquieres low fidelity training data from given databank (json file). This is especially designed to read out the 
    derivative of the coupling work requiered to insert a molecule of component i into a mixture of i and j.

    Args:
        settings (dict): Containing information about:
                            - "lf_databank": path to databank
                            - "lf_mixture": Mixture that should be utilized as low fidelity model (named: component1_component2)
                            - "lf_unique_key": Thermodynamic key for this mixture (e.g.: Temperature or pressure) 
                            - "lf_key": If vdW or coulomb portion of coupling work is investigated
                            - "charged": Boolean list if components posses partial charges (if not skip coulomb portion)

    Returns:
        l_learn (2D list): List containing lambda intermediates for every composition for every component
        xi_learn (2D list): List containing composition intermediates for every composition for every component
        du_dl_learn (2D list): List containing derivative of the coupling work for every composition for every component
    """
    
    l_learn     = [[],[]]
    xi_learn    = [[],[]]
    du_dl_learn = [[],[]]

    lf_dict     = work_json(settings["lf_databank"][settings["lf_key"]],to_do="read")

    key   = settings["lf_unique_key"][settings["lf_key"]]
    mix   = settings["lf_mixture"][settings["lf_key"]]
    comps = mix.split("_")

    for ni,comp in enumerate(comps):
        
        if not settings["charged"][ni] and settings["lf_key"] == "coulomb": continue

        l_learn[ni]     = [ np.array(item[1]["lambda"]) for item in lf_dict[mix][key][comp].items() ] 
        xi_learn[ni]    = [ np.ones(len(lf_dict[mix][key][comp][xi]["lambda"]))*float(xi) for i,xi in enumerate(lf_dict[mix][key][comp].keys()) ] 
        du_dl_learn[ni] = [ np.array(item[1]["du_dl"]) for item in lf_dict[mix][key][comp].items() ] 
            
    return l_learn,xi_learn,du_dl_learn

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


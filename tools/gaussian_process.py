import GPy
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class GPR():
    
    def __init__(self,X,Y,kernel=GPy.kern.Matern32,n_optimization_restarts=1) -> None:

        axis,axis_inv = 0,-1

        ## Prepare input data ##

        # Safe input data in class #
        self.X        = X
        self.Y        = Y
        
        # Normalize data (frobenius norm for each input dimension) #
        self.X_normer = np.linalg.norm(self.X,axis=axis)
        self.Y_normer = np.linalg.norm(self.Y,axis=axis)       
        
        self.X_norm   = self.X/self.X_normer
        self.Y_norm   = self.Y/self.Y_normer
        
        self.X_min    = np.min(self.X,axis=axis)
        self.X_max    = np.max(self.X,axis=axis)

        ## Setup Gaussian process regression model ##

        self.dims     = self.X.shape[axis_inv]
        kernels       = kernel(self.dims) if self.dims==1 else kernel(self.dims,ARD=True) 

        self.model    = GPy.models.GPRegression( self.X_norm, self.Y_norm, kernels )

        self.nrestart = n_optimization_restarts

    def train(self):
        """
        Train the Gaussian process object.
        
        Adjust and fix hyperparameters between initialisation and training of the model
        """
        self.model.optimize_restarts(num_restarts=self.nrestart)

        return

    def predict(self,x):
        """
        Predicts un-normed results for a trained Gaussian process model
        
        x (2D array): (un-normed) evaluation locations, can be multi dimensional (by appending more columns )
            
        returns mean predictions and corresponding variances as np.arrays
        """        
        X  = x/self.X_normer

        predict_mean, predict_var = self.model.predict(X)

        return predict_mean*self.Y_normer, predict_var*self.Y_normer**2 
    
    def plot(self):
        return

def get_hf_prediction( component: str, hf_data: List[List],dimension: int=2, x_pred: np.ndarray=np.array([]),  
                       lengthscale: List[float]=[], fix_lengthscale: bool=False, fix_hf_noise: bool=True, 
                       no_lambda_intermediates: int=51, verbose=False) -> Tuple[ np.ndarray, np.ndarray, np.ndarray ]:
    """
    Function that uses a GPR to interpolate dH/dlambda from free energy simulations. 
    This will loop through every composition of the mixture and return the interpolated dH/dlambda as well as its variance for the specified component.


    Args:
        component (str): Component for which the multi fidelity modeling is performed.
        hf_data (list): List containing several sublists. In the first sublist, the lambda training points are provided for every composition of component.
                        In the 2nd sublist, the composition training points are provided for every composition of component.
                        In the 3th sublist, the du_dlambda training data is provided for every composition of component.
                        (Optional) In the 4th sublist, the variance of the du_dlambda training data is provided for every composition of component.
        dimension (int, optional): Dimension of the multi fidelity modeling. 2D includes the compositions, 1D is just dH/dl over lambda. Defaults to 2.
        x_pred (np.ndarray, optional): New evaluated liquid compositions (in case of 2D multi fidelity). Defaults to np.array([]).
        lengthscale (List[float], optional): Possible default lenghtscale hyperparameters. Defaults to [].
        fix_lengthscale (bool, optional): If the lenghtscale hyperparameters should be fixed. Defaults to False.
        fix_hf_noise (bool, optional): If the noise of the high fidelity should be fixed. Defaults to True.
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

        print(f"\n2D GPR modeling for component: {component}\n")

        # Prepare 2D data
        X_train = np.concatenate( [np.column_stack( ( _ ) ) for _ in zip( hf_data[0], hf_data[1] ) ] )
        Y_train = np.concatenate( (hf_data[2]) ).reshape(-1,1)

        # Train GPR model 
        gpr_modeling = GPR(X_train, Y_train, n_optimization_restarts = 3)

        # Set initial values (might results in faster and better convergence of hyperparameter optimization)
        if bool(lengthscale):
            gpr_modeling.model.Mat32.lengthscale   = lengthscale

        # If wanted constrain hyperparameters 
        if fix_lengthscale:
            gpr_modeling.model.Mat32.lengthscale.fix()

        # Use high fidelity simulation variance to fix noise of high fidelity
        if len(hf_data) == 4 and fix_hf_noise:
            gpr_modeling.model.Gaussian_noise.variance = np.mean( hf_data[3] ) / gpr_modeling.Y_normer**2
            gpr_modeling.model.Gaussian_noise.variance.fix()
        
        # Optimize hyperparemeters
        gpr_modeling.train()

        if verbose:
            print("High fidelity variance: %.3f"%gpr_modeling.model[0])
            print("High fidelity lengthscales: %.3f, %.3f"%(gpr_modeling.model[1],gpr_modeling.model[2]))
            print("High fidelity noise: %.3f"%gpr_modeling.model[3])
        
        # Acquire high fidelity prediction
        col1, col2     = np.meshgrid(l_pred, x_pred)
        X_pred         = np.column_stack((col1.ravel(), col2.ravel()))

        hf_mean,hf_var = gpr_modeling.predict(X_pred)

    else:

        print(f"\n1D GPR for component: {component}\n")
        hf_mean,hf_var = [],[]

        # *hf_var_du_dl syntax is used to capture the remaining elements in a list. If none are left, then hf_var_du_dl will be empty
        for hf_l, hf_x, hf_du_dl, *hf_var_du_dl in zip( *hf_data ):

            # Prepare multi fidelity data as 1D data
            X_train = np.array(hf_l).reshape(-1,1)
            Y_train = np.array(hf_du_dl).reshape(-1,1)

            # Train GPR model
            gpr_modeling = GPR(X_train, Y_train, n_optimization_restarts = 3)

            # Set initial values (might results in faster and better convergence of hyperparameter optimization)
            if bool(lengthscale):
                gpr_modeling.model.Mat32.lengthscale   = lengthscale[0]

            # If wanted constrain hyperparameters 
            if fix_lengthscale:
                gpr_modeling.model.Mat32.lengthscale.fix()

            # Use high fidelity simulation variance to fix noise of high fidelity
            if len(hf_data) == 4 and fix_hf_noise:
                gpr_modeling.model.Gaussian_noise.variance = np.mean( hf_var_du_dl ) / gpr_modeling.Y_normer**2
                gpr_modeling.model.Gaussian_noise.variance.fix()

            # Optimize hyperparemeters
            gpr_modeling.train()

            if verbose:
                print("High fidelity variance: %.3f"%gpr_modeling.model[0])
                print("High fidelity lengthscale: %.3f"%gpr_modeling.model[1])
                print("High fidelity noise: %.3f"%gpr_modeling.model[2])

            ## Acquire high fidelity prediction ##

            X_pred    = l_pred.reshape(-1,1)
            hf_m,hf_v = gpr_modeling.predict(X_pred)

            hf_mean.append(hf_m)
            hf_var.append(hf_v)

        # Concatenate all high fidelity predictions for every high fidelity composition
        hf_mean, hf_var = np.concatenate(hf_mean), np.concatenate(hf_var)
        
    return l_pred, hf_mean, hf_var
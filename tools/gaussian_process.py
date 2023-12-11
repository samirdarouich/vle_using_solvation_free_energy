import GPy
import numpy as np
import matplotlib.pyplot as plt

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

        hf_mean, hf_var = self.model.predict(X)

        return hf_mean*self.Y_normer, hf_var*self.Y_normer**2 
    
    def plot(self):
        return
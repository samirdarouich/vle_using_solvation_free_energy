
import numpy as np
from typing import List, Tuple

class naturalcubicspline():
    """
    Class that performes natural cubic spline interpolation.

    self.w_k are the weights used in an integration scheme
    """
    def __init__(self, x):

        # define some space
        L = len(x)
        H = np.zeros([L,L],float)
        M = np.zeros([L,L],float)
        BW = np.zeros([L,L],float)
        AW = np.zeros([L,L],float)
        DW = np.zeros([L,L],float)

        h = x[1:L]-x[0:L-1]
        ih = 1.0/h

        # define the H and M matrix, from p. 371 "applied numerical methods with matlab, Chapra"
        H[0,0] = 1
        H[L-1,L-1] = 1
        for i in range(1,L-1):
            H[i,i] = 2*(h[i-1]+h[i])
            H[i,i-1] = h[i-1]
            H[i,i+1] = h[i]

            M[i,i] = -3*(ih[i-1]+ih[i])
            M[i,i-1] = 3*(ih[i-1])
            M[i,i+1] = 3*(ih[i])

        CW = np.dot(np.linalg.inv(H),M)  # this is the matrix translating c to weights in f.
                                                    # each row corresponds to the weights for each c.

        # from CW, define the other coefficient matrices
        for i in range(0,L-1):
            BW[i,:]    = -(h[i]/3)*(2*CW[i,:]+CW[i+1,:])
            BW[i,i]   += -ih[i]
            BW[i,i+1] += ih[i]
            DW[i,:]    = (ih[i]/3)*(CW[i+1,:]-CW[i,:])
            AW[i,i]    = 1

        # Make copies of the arrays we'll be using in the future.
        self.x  = x.copy()
        self.AW = AW.copy()
        self.BW = BW.copy()
        self.CW = CW.copy()
        self.DW = DW.copy()

        # find the integrating weights
        self.wsum = np.zeros([L],float)
        self.wk = np.zeros([L-1,L],float)
        for k in range(0,L-1):
            w = DW[k,:]*(h[k]**4)/4.0 + CW[k,:]*(h[k]**3)/3.0 + BW[k,:]*(h[k]**2)/2.0 + AW[k,:]*(h[k])
            self.wk[k,:] = w
            self.wsum += w

    def interpolate(self, y: List[float], xnew: List[float]):
        """
        Function that interpolates the provided y values at xnew points

        Args:
            y (List[float]): Provided values that should be interpolated.
            xnew (List[float]): Interpolation locations.

        Returns:
            _type_: _description_
        """
        if len(self.x) != len(y):
            print("\nThe length of 'y' should be consistent with that of 'self.x'. I cannot perform linear algebra operations.")
        # get the array of actual coefficients by multiplying the coefficient matrix by the values
        a = np.dot(self.AW,y)
        b = np.dot(self.BW,y)
        c = np.dot(self.CW,y)
        d = np.dot(self.DW,y)

        N = len(xnew)
        ynew = np.zeros([N],float)
        for i in range(N):
            # Find the index of 'xnew[i]' it would have in 'self.x'.
            j = np.searchsorted(self.x, xnew[i]) - 1
            lamw = xnew[i] - self.x[j]
            ynew[i] = d[j]*lamw**3 + c[j]*lamw**2 + b[j]*lamw + a[j]
            
        # Preserve the terminal points.
        ynew[0] = y[0]
        ynew[-1] = y[-1]

        return ynew

def trapezoid_integration(x: List[float], y: List[float], y_var: List[float]=[]):
    """
    Function that perform trapezoid integration. If provided, error propagation through integration is performed 
    and the resulting standard deviation is returned

    Args:
        x ( List[float]): Data points
        y ( List[float]): Data values
        y_var ( List[float], optional): Data variance. Defaults to [].

    Returns:
        Y (float): Integral value
        Y_std (float): Standard deviation of integral value
    """
    
    weights = np.array( [(x[1]-x[0])/2] + [(x[i+1]-x[i-1])*0.5 for i,_ in enumerate(x) if not (i==0 or i==len(x)-1)] + [(x[-1]-x[-2])/2] )

    Y     = np.dot( weights, y )
    Y_std = np.sqrt( np.dot( weights**2, y_var ) ) if len(y_var)==len(x) else 0.0

    return Y, Y_std

def cubic_integration(x: List[float], y: List[float], y_var: List[float]=[]):
    """
    Function that perform cubic integration. If provided, error propagation through integration is performed 
    and the resulting standard deviation is returned

    Args:
        x ( List[float]): Data points
        y ( List[float]): Data values
        y_var ( List[float], optional): Data variance. Defaults to [].

    Returns:
        Y (float): Integral value
        Y_std (float): Standard deviation of integral value
    """
    
    ncs     = naturalcubicspline(np.array(x))
    weights = ncs.wk.copy()
    
    # Cubic spline weights are a matrix, in each element k (weights[k,:]) are the integration weights to integrate from point k to k+1. The integration weights are for each point.
    # Hence, if one integrates from k to k+1, one still consider all other functions values with there corresponding weight.
    # The final integral is the summation over each subintegral.

    Y     = np.sum( np.dot( weights, y ) )
    Y_std = np.sqrt( np.sum( np.dot( weights**2, y_var ) ) ) if len(y_var)==len(x) else 0.0

    return Y, Y_std

def fit_poly(x: List[float], y: List[float], x_eval: List[float], deg: int=2, w: List[float]=None) -> Tuple[ np.ndarray, np.ndarray]:
    """
    Function that performs a polynomial fit to given data and evaluate it at specified locations
    
    Args:
        x (List[float]): Locations of data points
        y (List[float]): Data for given locations
        x_eval (List[float]): Evaluation locations of fitted polynomial
        deg (int,optional): Degree of polynomial. Defaults to 2.
        weights (List[float], optional): List of fittings weights for each residuum. Defaults to None.
    
    Returns:
        y_eval (np.ndarray): Data points of fitted polynomial to given locations
        std (np.ndarray): Corresponding uncertanty due to polynomial fit and provided variances
    
    """
    
    # Error propagation in polynomial least square fit (https://iopscience.iop.org/article/10.1088/0026-1394/48/1/002/pdf)
    
    X = np.array( [ [ xi**d for d in range(deg+1) ] for xi in x_eval ] )

    # If w in not provided just set it to ones
    if w is None: w = np.ones(len(x))

    # Search for variance of given points to add them to the overall variance of the model
    add_variance = np.array( [ 1 / w[ list(x).index(xi) ] if xi in x else 0.0 for xi in x_eval ] )

    if len(x) <= deg+1:
        # To less datapoints for covariance matrix, only use input noise
        a, cov = np.polyfit( x, y, deg=deg, w=w ), np.zeros( (deg+1,deg+1) )
    else:
        # Evaluate the coefficients and covariance matrix of them
        a, cov = np.polyfit( x, y, deg=deg, cov=True, w=w )

    # Evaluate fitted y values, as well as there covariance (combined from model insecrurity and provided variance)
    y_eval = np.poly1d( a )( x_eval )
    std    = np.sqrt( np.diag( X.dot( np.flip(cov) ).dot( X.T ) ) + add_variance )

    return y_eval, std
#------------------------------------------------------------------------------
# numerical_integration.py
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# import necessary libraries
#------------------------------------------------------------------------------

import numpy as np 

#------------------------------------------------------------------------------
# functions 
#------------------------------------------------------------------------------

def simpson(f, x_init: float, x_final: float, n_intervals: int)  -> float:
    """

    Uses the Simpson's rule for numerical integration of
    some analytic or numerical function.
    
    Arguments:
    ----------
    f : function
        a python functions containing the mathematical 
        expression for integration
    x_init : float
        the initial point of the integration interval
    x_final : float 
        the final point of the integration interval
    n_intervals : int
        number of intervals
        
    Returns:
    --------
    int_sum : float
        the result of the numerical integration
    
    """

    assert ( n_intervals % 2 == 0 ), "Number of intervals must be an even number. " 

    int_sum = 0.0
    dx = (x_final - x_init) / n_intervals
    
    # integration of an analytical function
    if callable(f):
        for j in range(n_intervals):

            x1, x2 = j*dx, (j+1)*dx
            f1, f2 = f(x1), f(x2)
            f12 = f( (x1+x2)/2.0 )
            pre_factor = (x2-x1)/6.0
            term = pre_factor * ( f1 + 4.0 * f12 + f2 )
            int_sum += term
            
    # integration of a numerical function
    elif isinstance(f, np.ndarray) or isinstance(f, list):
        pre_factor = dx / 3.0
        for j in range(1, n_intervals - 2, 2):
            f1, f2 = f[j], f[j+1]
            term = pre_factor * ( 4.0 * f1 + 2.0 * f2 )
            int_sum += term
        int_sum += pre_factor * ( f[0] + f[-1] + 4.0 * f[-2] ) 

    return int_sum

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def trapezoidal(f, dx : float):
    """

    Integration of a function using the trapezoidal method.
    
    Parameters:
    ----------- 
    f : np:ndarray
        should be a row/column vector or a series of column vectors 
    dx : float
        interval between points 

    """
    
    # assert if f is not an ndarray type - temporary 
    err_msg = "Input data for integration must be discrete"
    assert isinstance(f, np.ndarray), err_msg 

    # check if f is a 1D or 2D array
    sh = f.shape
    if len(sh) == 1:
        f = np.reshape( f, (1, sh[0]) )
    elif len(sh) > 2:
        raise Exception("Only 1D or 2D matrices are allowed.") 
     
    # create f[j+1]   
    f_for = np.roll( f, -1 )
    
    # evaluate (f[j+1] + f[j]) - skip the last column
    f_sum = (f[:,:-1] + f_for[:,:-1]) 
    factor = 2.0 / dx 
    
    # "integrate"
    res = np.sum( f_sum / factor, axis=1 ) 
    return res

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def trapezoidal3D(f, dx):
    """
    Integration of a LxL matrix over all elements. 

    Parameters:
    -----------
    f : np.array
        a LxLxN matrix, where the integration is carried out in the 3rd direction
        (axis=2)
    dx : float
        the size of the interval

    Returns:
    --------
    res : np.array
        a LxL array of the results of the integration

    """

    # assert if the number of dimensions of the array f is not 3
    assert len(f.shape) == 3, "Invalid number of dimensions. Use 3."  

    prefactor = dx / 2.0
    f_shifted = np.roll( f, -1, axis=0) 
    df = f[0:-1] + f_shifted[0:-1]  
    res = np.sum(df, axis=0) * prefactor 
    return res

#------------------------------------------------------------------------------
# end of .py file
#------------------------------------------------------------------------------

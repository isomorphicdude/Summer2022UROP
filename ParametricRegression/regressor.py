"""Implements a regressor function."""  

import numpy as np
from tqdm import tqdm


from solvers.euler import *
from solvers.rk_solvers import *
from ParametricRegression.models_for_regression import *
from solvers.gn_solver import GaussNewtonSolver  


def regressor(x, y, t, model,
                init_guess, 
                solver, 
                solver_params, 
                fit_params):

    """
    Returns the optimal parameters given data and model.    

    Args:  
        - x: array-like, inputs, shape (num_data, 2*(2N+1)), 2 dimensional  
                could also have shape (num_data, 2*(N+1)). if velocity is not given  
        - y: array-like, true values, shape (num_data, 2n*(2N+1)), 2 dimensional  
                could also have shape (num_data, 2n*(N+1)). if velocity is not given  
        - t: float, descent step size
        - model: function, model to be fitted
        - init_guess: array-like, initial guess for parameters
        - solver: string, name of solver to use, default GaussNewton
        - solver_params: dict, parameters for the solver
        - fit_params: dict, parameters for fitting  

    Output:  
        - params: array-like, optimal parameters
    """   

    if solver == 'GaussNewton':
        print(f"Using {solver} solver")
        solver = GaussNewtonSolver(model, **solver_params)

    else:
        raise ValueError('Solver not implemented.')

    num_data = len(x)

    for i in tqdm(range(num_data), colour='green'):
        init_guess = solver.fit(x[i], y[i], t, init_guess, **fit_params)

    return init_guess


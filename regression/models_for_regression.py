"""Implements models for regression."""  

import numpy as np
from tqdm import tqdm

from models import *
from solvers.rk_solvers import rk2



def fModel4(x, vec):
    """
    Returns a flattened output.  

    Args:  
        - x: ndarray, initial conditions
        - vec: ndarray, parameters of length 3
    
    """  
    params = {'no. of prey': 100, 
    'kappa for prey': 0.5, 
    'attraction of prey a': vec[0], 
    'repulsion of prey b_1': vec[1], 
    'repulsion of pred b_2': vec[2],
    'p1 spotted': 0.8,
    'p2 not spotted':0.4,
    'angle_prey': np.cos(np.pi / 3),
    'angle_pred': np.cos(np.pi / 2.5), 
    'num_neighbours': 5,
    'attraction of pred c': 10, 
    'exponent of dist pred p': 3}

    N = 100
    times = (0,20)
    steps = 1000
    start, end = times  
    h = (end - start) / steps
    x = x.reshape(-1, 2) # reshape to coordinate form of initial conditions
    y = rk2(model4, x, h, 2*N+1, times, params, return_vel=False, verbose=False)

    return np.ravel(y)
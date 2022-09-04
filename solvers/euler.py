'''Implements Euler's method.'''

import numpy as np  
from tqdm import tqdm


def feuler(f, c0, h, dim, times, params, return_vel = True, verbose = True):
  '''
  Implements forward euler for 2D coordinates.   

  Parameters:  

    - f: function(y, params), the RHS of the DE, returning (N+1, 2) ndarray of derivatives
    - c0: ndarray of shape (dim, 2), initial condition 
    - h: float, time-step size
    - dim: int, dimension of the prolem, used to initialize array for storage
    - times: tuple, (start, end)
    - params: parameters (N,a,b,c,p)
    - return_vel: bool, indicates whether to return velocity of predator, default to True
    - verbose: bool, indicates whether to print progress, default to True

  Returns:
    - y: ndarray of values of shape computed at given times
    - extra: ndarray of extra results (e.g. the velocities)
  '''  
  
  start, end = times  
  t_span = np.arange(start, end, h)

  # length of time
  t_len = len(t_span)
  if verbose:
    print(f"No. of time steps: {t_len}")

  # initialize array to store values time x dimension x 2
  y = np.zeros((t_len, dim, 2))
  if return_vel:
    extra = np.zeros((t_len, 2))

  y[0] = c0    

  iter = tqdm(range(t_len-1)) if verbose else range(t_len-1)

  if return_vel:
    for i in iter:
      grad = f(y[i], params)
      extra[i] = grad[-1]
      y[i+1] = y[i] + h * grad

      if verbose and i%100 == 0:
        print(f"Iteration no. {i}")
    out = (y, extra)

  else:
    for i in iter:
      grad = f(y[i], params)
      y[i+1] = y[i] + h * grad

      if verbose and i%100 == 0:
        print(f"Iteration no. {i}")
    out = y  

  return out
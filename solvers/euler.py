'''Implements Euler's method.'''

import numpy as np  
from tqdm import tqdm


def feuler(f, c0, h, dim, times, params, verbose = False,
          return_vel = True):
  '''
  Implements forward euler for 2D coordinates.   

  Parameters:  

    - f: function(y, params), the RHS of the DE, returning (N+1, 2) ndarray of derivatives
    - c0: ndarray of shape (dim, 2), initial condition 
    - h: float, time-step size
    - dim: int, dimension of the prolem, used to initialize array for storage
    - times: tuple, (start, end)
    - params: parameters (N,a,b,c,p)
    - verbose: bool, indicates whether to print progress (for tracking)  
    - return_vel: bool, indicates whether to return velocity of predator

  Returns:
    - y: ndarray of values of shape computed at given times
    - extra: ndarray of extra results (e.g. the velocities)
  '''  
  
  start, end = times  
  t_span = np.arange(start, end, h)

  # length of time
  t_len = len(t_span)
  print(f"No. of time steps: {t_len}")

  # initialize array to store values time x dimension x 2
  y = np.zeros((t_len, dim, 2))
  if return_vel:
    extra = np.zeros((t_len, 2))

  y[0] = c0  
  # a list storing the derivatives for debugging
  # grads = [] 

  if return_vel:
    for i in tqdm(range(t_len-1)):
      grad = f(y[i], params)
      extra[i] = grad[-1]
      # grads.append(grad)
      y[i+1] = y[i] + h * grad

      if verbose and i%100 == 0:
        print(f"Iteration no. {i}")
    out = (y, extra)

  else:
    for i in tqdm(range(t_len-1)):
      grad = f(y[i], params)
      # grads.append(grad)
      y[i+1] = y[i] + h * grad

      if verbose and i%100 == 0:
        print(f"Iteration no. {i}")
    out = y  

  # if derivatives become constant
  # if verbose:
  #   for j in range(t_len-2):
  #     if (grads[j] == grads[j+1]).all():
  #       print(j)
  #       break
  return out
'''Implements Euler's method.'''

import numpy as np  
from tqdm import tqdm


def feuler(f, c0, h, times, params, verbose = True):
  '''
  Implements forward euler.  

  Parameters:
    - f: function(time, u), the RHS of the DE, returning (dim, 2) ndarray of derivatives
    - c: ndarray of shape (dim, 2), initial condition 
    - h: float, time-step size
    - times: tuple, (start, end)
    - params: parameters (N,a,b,c,p)
    - verbose: bool, indicates whether to print progress (for tracking)
  Returns:
    - y: ndarray of values computed at given times
  '''  

  start, end = times  
  t_span = np.arange(start, end, h)
  N, a, b, c, p = params

  # length of time
  t_len = len(t_span)
  print(f"No. of time steps: {t_len}")
  # initialize array to store values time x dimension x 2
  y = np.zeros((t_len, N+1, 2))

  y[0] = c0  
  # a list storing the derivatives for debugging
  grads = [] 

  for i in tqdm(range(t_len-1)):
    grad = f(y[i][0:N], y[i][-1:], a, b, c, p)
    grads.append(grad)
    y[i+1] = y[i] + h * grad
    if verbose and i%100 == 0:
      print(f"Iteration no. {i}")
  # if derivatives become constant
  # if verbose:
  #   for j in range(t_len-2):
  #     if (grads[j] == grads[j+1]).all():
  #       print(j)
  #       break
  return y  
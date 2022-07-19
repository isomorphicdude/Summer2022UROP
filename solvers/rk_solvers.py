'''Implements runge-kutta methods.'''  

import numpy as np
from tqdm import tqdm

def rk2(f, y0, h, times, params):
  '''
  Implements 2nd order Runge-Kutta.  

  Parameters:
  - f: function(time, u), the RHS of the DE, returning (dim, 2) ndarray of derivatives
  - y0: ndarray of shape (dim, 2), initial condition 
  - h: float, time-step size
  - times: tuple, (start, end)
  - params: parameters
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

  y[0] = y0  

  for i in tqdm(range(t_len-1)):
    k1 = np.array(f(y[i][0:N], y[i][-1:], a, b, c, p))
    y1 = y[i] + (h/2) * k1
    k2 = f(y1[0:N], y1[-1:], a, b, c, p)
    y[i+1] = y[i] + h * k2
    
  return y  



def rk4(f, y0, h, times, params):
  '''
  Implements 4th order Runge-Kutta.  

  Parameters:
  - f: function(time, u), the RHS of the DE, returning (dim, 2) ndarray of derivatives
  - y0: ndarray of shape (dim, 2), initial condition 
  - h: float, time-step size
  - times: tuple, (start, end)
  - params: parameters
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

  y[0] = y0   
  
  for i in tqdm(range(t_len-1)):
    k1 = np.array(f(y[i][0:N], y[i][-1:], a, b, c, p))
    y1 = y[i] + (h/2) * k1
    k2 = np.array(f(y1[0:N], y1[-1:], a, b, c, p))
    y2 = y[i] + (h/2) * k2
    k3 = np.array(f(y2[0:N], y2[-1:], a, b, c, p))
    y[i+1] = y[i] + h * k3
    
  return y

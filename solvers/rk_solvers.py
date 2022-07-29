'''Implements runge-kutta methods.'''  

import numpy as np
from tqdm import tqdm

def rk2(f, y0, h, dim, times, params):
  '''
  Implements 2nd order Runge-Kutta.   

  Parameters:  

  - f: function(y, params), the RHS of the DE, returning (dim, 2) ndarray of derivatives
  - y0: ndarray of shape (dim, 2), initial condition 
  - h: float, time-step size
  - dim: int, dimension of the prolem, used to initialize array for storage
  - times: tuple, (start, end)
  - params: parameters  

  Returns:  
  - y: ndarray of values computed at given times
  '''  
  start, end = times  
  t_span = np.arange(start, end, h)

  # length of time
  t_len = len(t_span)
  print(f"No. of time steps: {t_len}")
  # initialize array to store values time x dimension x 2
  y = np.zeros((t_len, dim, 2))

  y[0] = y0  

  for i in tqdm(range(t_len-1)):  
    k1 = f(y[i], params)
    y1 = y[i] + (h/2) * k1
    k2 = f(y1, params)
    y[i+1] = y[i] + h * k2
    
  return y  



def rk4(f, y0, h, dim, times, params):
  '''
  Implements 4th order Runge-Kutta.  

  Parameters:  

  - f: function(y, params), the RHS of the DE, returning (dim, 2) ndarray of derivatives
  - y0: ndarray of shape (dim, 2), initial condition 
  - h: float, time-step size
  - dim: int, dimension of the prolem, used to initialize array for storage
  - times: tuple, (start, end)
  - params: parameters  

  Returns:  
  - y: ndarray of values computed at given times
  '''  
  start, end = times  
  t_span = np.arange(start, end, h)

  # length of time
  t_len = len(t_span)
  print(f"No. of time steps: {t_len}")
  # initialize array to store values time x dimension x 2
  y = np.zeros((t_len, dim, 2))

  y[0] = y0   
  
  for i in tqdm(range(t_len-1)):
    k1 = f(y[i], params)
    k2 = f(y[i] + (h/2) * k1, params)
    k3 = f(y[i] + (h/2) * k2, params)
    k4 = f(y[i] + h * k3, params)
    k = (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)
    y[i+1] = y[i] + h * k

  return y

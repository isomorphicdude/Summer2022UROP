'''Implements some auxiliary functions.'''  

import numpy as np
import matplotlib.pyplot as plt

def expInit(N = 100,
             style = 'ring',
             second_order = True,
             seed = 0):  
  '''  
  Initializes prey and predator.
  
  Input:  
    - N: int, the no. of prey, default 100   
    - style: string, 'ring' or 'random' initialization  
    - second_ord: bool, return a second order with velocity or not,
                  if true, return with no velocity of predator
    - seed: int, to initialize

  Output:  
    - c0: (N+1, 2) ndarray if second_order is set to False, else 
          (2N+2,2) ndarray with (position of prey, velocity of prey, 
          position of predator, velocity of predator)
  '''  
  if style == 'ring':

    np.random.seed(seed)
    unit_interval = np.arange(0, 1, 1e-4)
    angles = 2 * np.pi * np.random.choice(unit_interval, size = N, replace = False)  
    R = np.random.uniform(0.25, 0.5, size = N)
    xprey = np.cos(angles) * R + 0.5
    yprey = np.sin(angles) * R + 0.5
    xpred = 0.5
    ypred = 0.5  
    coord_prey = np.vstack((xprey, yprey)).T

    if second_order:
      out = np.vstack((coord_prey, 
                       np.zeros_like(coord_prey), 
                       [xpred, ypred], 
                       [0, 0]))  
    else:
      out = np.vstack((coord_prey, 
                       np.zeros_like(coord_prey), 
                       [xpred, ypred] 
                       ))  
      
  elif style == 'random':  

    np.random.seed(seed)
    unit_interval = np.arange(0, 1, 1e-4) # values of x,y coordinates
    init = np.random.choice(unit_interval, size = 2*N+2, replace = False)
    xprey = init[0:N]  
    yprey = init[N+1:-1]
    xpred = init[N]
    ypred = init[-1]
    coord_prey = np.vstack((xprey, yprey)).T  

    if second_order:
      out = np.vstack((coord_prey, 
                       np.zeros_like(coord_prey), 
                       [xpred, ypred], 
                       [0, 0]))  
    else:
      out = np.vstack((coord_prey, 
                       np.zeros_like(coord_prey), 
                       [xpred, ypred] 
                       )) 
  return out  

def multiPlot(case,
              axis_lim = None,
              second_order = True):
  '''
  Plots diagram given sampling times.  
  
  Parameters:
    - case: a tuple containing the following
      - soln: ndarray, solution  
      - h: time step size
      - sample_points: list, times to plot
      - size: int, fig size
      - N: no. of prey   
    - axis_lim: int, set limits of axes to [-n, n]
    - second_order: bool, return a second order with velocity or not,
                    if true, return with velocity zero   
  '''  
  soln, h, sample_points, size, N = case
  n = len(sample_points)
  plt.figure(figsize=(size, size))  

  for j in range(n):  

    s = int(np.sqrt(n))+1
    ax = plt.subplot(s, s, j + 1)  
    i = int(sample_points[j] / h)
    i = i-1 if i>0 else 0

    if second_order:
      plt.scatter(soln[i][0:N, 0], soln[i][0:N, 1])
      plt.scatter(soln[i][2*N, 0], soln[i][2*N, 1])

      if axis_lim:
        plt.xlim([-1 * axis_lim, axis_lim])
        plt.ylim([-1 * axis_lim,axis_lim])

      plt.title(f'Time {sample_points[j]}')

    else:
      plt.scatter(soln[i][0:N, 0], soln[i][0:N, 1])
      plt.scatter(soln[i][-1, 0], soln[i][-1, 1])

      if axis_lim:
              plt.xlim([-1 * axis_lim, axis_lim])
              plt.ylim([-1 * axis_lim,axis_lim])

      plt.title(f'Time {sample_points[j]}')
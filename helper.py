'''Implements some auxiliary functions.'''  

import numpy as np
import matplotlib.pyplot as plt

def ringInit(N, seed = 0):  
  '''Initializes prey in a ring.'''  
  np.random.seed(seed)
  unit_interval = np.arange(0, 1, 1e-4)
  angles = 2 * np.pi * np.random.choice(unit_interval, size = N, replace = False)  
  R = np.random.uniform(0.25, 0.5, size = N)
  xprey = np.cos(angles) * R + 0.5
  yprey = np.sin(angles) * R + 0.5
  xpred = 0.5
  ypred = 0.5
  c0 = np.vstack((np.vstack((xprey, yprey)).T, [xpred, ypred]))
  return c0  

def randInit(N, seed = 0):
  '''Initializes prey randomly.'''  
  np.random.seed(seed)
  unit_interval = np.arange(0, 1, 1e-4) # values of x,y coordinates
  init = np.random.choice(unit_interval, size = 2*N+2, replace = False)
  xprey = init[0:N]  
  yprey = init[N+1:-1]
  xpred = init[N]
  ypred = init[-1]
  c0 = np.vstack((np.vstack((xprey, yprey)).T, [xpred, ypred]))
  return c0  

def multiPlot(case):
  '''
  Plots diagram given sampling times.  
  
  Parameters:
    - case: a tuple containing the following
      - soln: ndarray, solution  
      - h: time step size
      - sample_points: list, times to plot
      - size: int, fig size
      - N: no. of prey
  '''  
  soln, h, sample_points, size, N = case
  n = len(sample_points)
  plt.figure(figsize=(size, size))
  for j in range(n):  
    s = int(np.sqrt(n))+1
    ax = plt.subplot(s, s, j + 1)  
    i = int(sample_points[j] / h)
    i = i-1 if i>0 else 0
    plt.scatter(soln[i][0:N, 0], soln[i][0:N, 1])
    plt.scatter(soln[i][-1, 0], soln[i][-1, 1])
    # plt.xlim([-2,2])
    # plt.ylim([-2,2])
    plt.title(f'Time {sample_points[j]}')
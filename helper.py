'''Implements some auxiliary functions.'''  

import os
import numpy as np
import matplotlib.pyplot as plt
from solvers.rk_solvers import *
from solvers.euler import *


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
              sample_points,
              size = 12,
              axis_lim = None,
              second_order = True,
              quiver = False):
  '''
  Plots diagram given sampling times.  
  
  Parameters:
    - case: a tuple containing the following   
      - soln: ndarray, solution of positions at different times
              or (y, pred_vel) for quiver plot
      - h: time step size
      - N: no. of prey   
    - sample_points: list, times to plot
    - size: int, fig size
    - axis_lim: int, set limits of axes to [-n, n]
    - second_order: bool, return a second order with velocity or not,
                    if true, return with velocity zero    
    - quiver: bool, determine if return quiver plot when velocity is provided
  '''  
  soln, h, N = case
  n = len(sample_points)
  plt.figure(figsize=(size, size))  

  try:
    soln, vel_pred = soln
  except ValueError:
    print("You are using just the positions.")

  for j in range(n):  

    s = int(np.sqrt(n))+1
    ax = plt.subplot(s, s, j + 1)  
    i = int(sample_points[j] / h)
    i = i-1 if i>0 else 0

    if second_order and not quiver:
      plt.scatter(soln[i][0:N, 0], soln[i][0:N, 1])
      plt.scatter(soln[i][2*N, 0], soln[i][2*N, 1])

      if axis_lim:
        plt.xlim([-1 * axis_lim, axis_lim])
        plt.ylim([-1 * axis_lim,axis_lim])

      plt.title(f'Time {sample_points[j]}')

    if second_order and quiver:
      plt.scatter(soln[i][0:N, 0], soln[i][0:N, 1])
      plt.quiver([soln[i][0:N, 0]], [soln[i][0:N, 1]],
                soln[i][N:2*N, 0], soln[i][N:2*N, 1])
      plt.scatter(soln[i][2*N, 0], soln[i][2*N, 1])
      plt.quiver([soln[i][2*N, 0]], [soln[i][2*N, 1]],
                vel_pred[i][0], vel_pred[i][1])
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

def exportPlot(case,
              sample_points,
              size = 4,
              axis_lim = None,
              second_order = True,
              quiver = True,
              export_dir = 'plots',
              fontsize = 40):
  '''
  Plots and exports diagram given sampling times.  
  
  Parameters:
    - case: a tuple containing the following   
      - soln: ndarray, solution of positions at different times
              or (y, pred_vel) for quiver plot
      - h: time step size
      - N: no. of prey  
    - sample_points: list, times to plot
    - size: int, figsize
    - axis_lim: int, set limits of axes to [-n, n]
    - second_order: bool, return a second order with velocity or not,
                    if true, return with velocity zero    
    - quiver: bool, determine if return quiver plot when velocity is provided
              default is True
    - export_dir: string, target directory for import, default is 'plots'
    - fontsize: int, fontsize of title, default 40
  '''  
  soln, h, N = case
  n = len(sample_points)
  fig, axes = plt.subplots(1,n, figsize = (n*size, size))
  # abs_path = os.path.abspath(__file__)
  # path = os.path.join(abs_path, export_dir)
  # os.mkdir(path)

  try:
    soln, vel_pred = soln
  except ValueError:
    print("You are using just the positions.")

  for j in range(n):  
    i = int(sample_points[j] / h)
    i = i-1 if i>0 else 0

    if second_order and not quiver:
      axes[j].scatter(soln[i][0:N, 0], soln[i][0:N, 1])
      axes[j].scatter(soln[i][2*N, 0], soln[i][2*N, 1])

      if axis_lim:
        axes[j].set_xlim([-1 * axis_lim, axis_lim])
        axes[j].set_ylim([-1 * axis_lim,axis_lim])

      axes[j].set_title(f'Time {sample_points[j]}', fontsize=fontsize)


    if second_order and quiver:
      axes[j].scatter(soln[i][0:N, 0], soln[i][0:N, 1])
      axes[j].quiver([soln[i][0:N, 0]], [soln[i][0:N, 1]],
                soln[i][N:2*N, 0], soln[i][N:2*N, 1])
      axes[j].scatter(soln[i][2*N, 0], soln[i][2*N, 1])
      axes[j].quiver([soln[i][2*N, 0]], [soln[i][2*N, 1]],
                vel_pred[i][0], vel_pred[i][1])
      if axis_lim:
        axes[j].set_xlim([-1 * axis_lim, axis_lim])
        axes[j].set_ylim([-1 * axis_lim,axis_lim])

      axes[j].set_title(f'Time {sample_points[j]}', fontsize=fontsize)

    else:
      axes[j].scatter(soln[i][0:N, 0], soln[i][0:N, 1])
      axes[j].scatter(soln[i][-1, 0], soln[i][-1, 1])

      if axis_lim:
        axes[j].set_xlim([-1 * axis_lim, axis_lim])
        axes[j].set_ylim([-1 * axis_lim,axis_lim])

      axes[j].set_title(f'Time {sample_points[j]}', fontsize=fontsize)  
  name = os.path.join(export_dir, f"{sample_points}.svg")
  plt.savefig(name)



def computeSoln(func, params, steps, times,
                second_order = False,
                init_sty = 'random', method = "rk4",
                return_vel = True
                ):
  '''
  Computes solution and prints parameters.  

  Parameters:
    - func: function(y, params), the RHS of the DE, returning (dim, 2) ndarray of derivatives
    - params: py dict, the parameter of function
    - steps: int, no. of time steps
    - times: tuple, (start, end)
    - second_order: bool, indicates second order, defualt to False
    - init_sty: string, initialization style, can be 'random' or 'ring', default 'random'
    - method: string, solver to use, can be 'feuler', 'rk2', or 'rk4'

  Returns:  
    - list containing solution
  '''    

  N = params['no. of prey']

  start, end = times  
  h = (end - start) / steps
  c0 = expInit(N, init_sty, second_order=second_order)

  if second_order:
    dim = 2*N + 2
  else:
    dim = 2*N + 1

  if method == 'rk4':  
    y = rk4(func, c0, h, dim, times, params)
  elif method == 'rk2':
    y = rk2(func, c0, h, dim, times, params)
  elif method == 'feuler':
    y = feuler(func, c0, h, dim, times, params, 
              return_vel=return_vel)

  return [y, h, N]
  

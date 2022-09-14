"""Implements a data generator."""

import os
import time
import numpy as np

from models.helper import *
from models.models import *
from solvers.euler import *
from solvers.rk_solvers import *

from multiprocessing import Pool


def generateData(model,
                num_data = 100,
                init_sty = 'random',
                times = (0, 20),
                params = {'no. of prey': 100, 
                        'kappa for prey': 0.5, 
                        'attraction of prey a': 1, 
                        'repulsion of prey b_1': 1, 
                        'repulsion of pred b_2': 0.07, 
                        'attraction of pred c': 10, 
                        'exponent of dist pred p': 1.2},
                steps = 1000,
                second_order = False,
                method = 'rk2',
                return_vel = False,
                cores = 2,
				flattened = False
                ):
	"""  
	Returns a tuple of the shape (2N+n+1, 2n(2N+1)).   

	Args:    
	- model: the model to use  
	- num_data: int, no. of data points to generate  
	- init_sty: str, 'random' or 'ring', default is 'random'  
	- times: tuple, (start, end)  
	- params: dict, parameters  
	- steps: int, no. of time steps  
	- second_order: bool, indicates whether to use second order predator, default False
	- method: str, method to use  
	- return_vel: bool, return velocity or not, default to False  
	- cores: int, no. of cores to use    
	- flattened: bool, indicates whether to return value as a one-dim vector  

	Returns:    
	- data: list of tuples, ((initial_val, times), (pos_prey, vel_prey, pos_pred)), 
			when flattened=True, each element in the tuple is flattened  
	"""  

	# unpack parameters
	N = params['no. of prey']
	if cores>os.cpu_count():
		raise ValueError("No. of cores cannot be greater than no. of available cores.")

	chunksize = int(num_data/(cores*10)) if int(num_data/(cores*10))>0 else 1

	# initialize 
	start, end = times  
	h = (end - start) / steps

	# use a list as constant time for accessing & appending
	args = [] # list of arguments for mapping the function
	init_vals = [] # initial values 
	
	# determines if it's second order
	if second_order:
		dim = 2*N + 2
	else:
		dim = 2*N + 1

	for i in range(num_data):
		c0 = expInit(N, init_sty, second_order=second_order, seed=None)
		init_vals.append(c0)
		tup = (model, c0, h, dim, times, params, return_vel, False)
		args.append(tup)

	# multicore performance enhancement
	try:
		print("Trying to use multiprocessing...")
		with Pool(cores) as p:
			if method == 'rk4':  
				data = p.starmap(rk4, args, chunksize=chunksize)
			elif method == 'rk2':
				data = p.starmap(rk2, args, chunksize=chunksize)
			elif method == 'feuler':
				data = p.starmap(feuler, args, chunksize=chunksize)
			else:
				raise ValueError("Invalid method.")
		print("Multiprocessing successful.")

	except:
		print("Error in generating data using multiprocessing.")
		print("Switching to single core...")
		data = []
		for params in args:
			# preprocessing of solutions need to be done here
			if method == 'rk4':  
				soln = rk4(*params)
				# need to add time and init tuple as \mathbf{x}
				data.append(soln)
			elif method == 'rk2':
				soln = rk2(*params)
				data.append(soln)
			elif method == 'feuler':
				soln = feuler(*params)
				data.append(soln)
			else:
				raise ValueError("Invalid method.")

	if flattened:
		data = [np.ravel(val) for val in data]
		init_vals = [np.ravel(val) for val in init_vals]


	data = zip(init_vals, data)

	return list(data)



'''Collection of models.'''  

import numpy as np

def model0(y, params):
  '''
  Returns ndarray of shape (dim, 2) containing the derivatives
  Uses second order for prey.   
  
  Input:   
    - y: an array containing the following inputs   
      - pprey: ndarray of (N,2), the positions of the prey  
      - vprey: ndarray of (N,2), the velocities of the prey  
      - ppred: ndarray of (1,2), the positions of the predator  
      - vpred: ndarray of (1,2), the velocities of the predator  

    - params: extra parameters for the model (N, kh, kp, a, b1, b2, c, p) for now

  Returns:   
    - dy: ndarray of (2*N+2, 2), the derivatives of the inputs  
  '''  

  # unpacking parameters
  N, kh, kp, a, b1, b2, c, p = params 

  pprey = y[0:N]  
  vprey = y[N:2*N]  
  ppred = y[2*N:2*N+1]
  vpred = y[2*N+1:]  


  # List of len N, ndarrays of shape (N-1, 2)
  dist_prey =  np.array([pprey[i] - np.delete(pprey, i, axis = 0) for i in range(N)])  
  
  vel_prey  =  np.array([vprey[i] - np.delete(vprey, i, axis = 0) for i in range(N)])

  # kernel of prey
  ker_prey = kh * np.sum (-1 * vel_prey / (1 + (np.linalg.norm(dist_prey, axis = 2)[:,:,np.newaxis]**2))
                    ,axis = 1)   
  
  # attraction of prey
  attrac_prey = -1 * a * np.sum(dist_prey, axis = 1)  

  # repulsion of prey
  rep_prey = b1 * np.sum (dist_prey / (np.linalg.norm(dist_prey, axis = 2)[:,:,np.newaxis]**2),
                          axis = 1)
                          
  # repulsion of predator
  rep_pred = b2 * (pprey - ppred) / (np.linalg.norm(pprey - ppred, axis = 1)[:,np.newaxis]**2)  

  # acceleration of prey
  acc_prey = (1/N) * (ker_prey + attrac_prey + rep_prey) + rep_pred  

  # kernel of predator
  ker_pred = kp * np.sum((vprey - vpred) / (1 + np.linalg.norm(ppred - pprey, axis = 1)[:,np.newaxis]**2),
                    axis = 0)

  # acceleration of predator
  acc_pred = (1/N) * (ker_pred + 
                      c * np.sum((pprey - ppred) / (np.linalg.norm(pprey - ppred, axis = 1)[:,np.newaxis]**p), 
                                 axis = 0))[np.newaxis]
                                 
  return np.concatenate((vprey, acc_prey, vpred, acc_pred))
  
'''Collection of models.'''  

import numpy as np

def model0(y, params):
  '''
  Returns ndarray of shape (dim, 2) containing the derivatives
  with both predator and prey second order.    
  
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

  norm_prey =  np.linalg.norm(dist_prey, axis = 2)[:,:,np.newaxis]

  # kernel of prey
  ker_prey = kh * np.sum (-1 * vel_prey / (1 + (norm_prey**2))
                    ,axis = 1)   
  
  # attraction of prey
  attrac_prey = -1 * a * np.sum(dist_prey, axis = 1)  

  # repulsion of prey
  rep_prey = b1 * np.sum (dist_prey / (norm_prey**2),
                          axis = 1)
                          
  # repulsion of predator
  norm_pred = np.linalg.norm(pprey - ppred, axis = 1)[:,np.newaxis]
  rep_pred = b2 * (pprey - ppred) / (norm_pred**2)  

  # acceleration of prey
  acc_prey = (1/N) * (ker_prey + attrac_prey + rep_prey) + rep_pred  

  # kernel of predator
  ker_pred = kp * np.sum((vprey - vpred) / (1 + norm_pred**2),
                    axis = 0)

  # acceleration of predator
  acc_pred = (1/N) * (ker_pred + 
                      c * np.sum((pprey - ppred) / (norm_pred**p), 
                                 axis = 0))[np.newaxis]
                                 
  return np.concatenate((vprey, acc_prey, vpred, acc_pred))  



def model1(y, params):
  '''
  Returns ndarray of shape (dim, 2) containing the derivatives
  with prey second order and predator first order.     
  
  Input:   
    - y: an array containing the following inputs   
      - pprey: ndarray of (N,2), the positions of the prey  
      - vprey: ndarray of (N,2), the velocities of the prey  
      - ppred: ndarray of (1,2), the positions of the predator  

    - params: py dict, extra parameters for the model (N, kh, a, b1, b2, c, p) for now

  Returns:   
    - dy: ndarray of (2*N+1, 2), the derivatives of the inputs  
  '''  

  # unpacking parameters
  N, kh, a, b1, b2, c, p = params.values() 

  pprey = y[0:N]  
  vprey = y[N:2*N]  
  ppred = y[2*N:]
  # vpred = y[2*N+1:]  

  # ndarrays of shape (N, N-1, 2)
  dist_prey =  np.array([pprey[i] - np.delete(pprey, i, axis = 0) for i in range(N)])  
  
  vel_prey  =  np.array([vprey[i] - np.delete(vprey, i, axis = 0) for i in range(N)])  

  norm_prey =  np.linalg.norm(dist_prey, axis = 2)[:,:,np.newaxis]

  # kernel of prey
  ker_prey = kh * np.sum (-1 * vel_prey / (1 + (norm_prey**2))
                    ,axis = 1)   
  
  # attraction of prey
  attrac_prey = -1 * a * np.sum(dist_prey, axis = 1)  

  # repulsion of prey
  rep_prey = b1 * np.sum (dist_prey / (norm_prey**2),
                          axis = 1)
                          
  # repulsion of predator
  norm_pred = np.linalg.norm(pprey - ppred, axis = 1)[:,np.newaxis]
  rep_pred = b2 * (pprey - ppred) / (norm_pred**2)  

  
  # acceleration of prey
  acc_prey = (1/N) * (ker_prey + attrac_prey + rep_prey) + rep_pred  

  # velocity of predator
  vpred = (1/N) * (c * np.sum((pprey - ppred) / (norm_pred**p), 
                                 axis = 0))[np.newaxis]
                                 
  return np.concatenate((vprey, acc_prey, vpred))  



def model3(y, params):
  '''
  Returns ndarray of shape (dim, 2) containing the derivatives,
  with perception cone added.    
  
  Input:   
    - y: an array containing the following inputs   
      - pprey: ndarray of (N,2), the positions of the prey  
      - vprey: ndarray of (N,2), the velocities of the prey  
      - ppred: ndarray of (1,2), the positions of the predator  

    - params: py dict, extra parameters for the model 
              (N, kh, a, b1, b2, p1, p2, angle_prey, angle_pred, c, p)

  Returns:   
    - dy: ndarray of (2*N+1, 2), the derivatives of the inputs  
  '''  

  # unpacking parameters
  N, kh, a, b1, b2, p1, p2, a_h, a_p, c, p = params.values()
  # N, kh, a, b1, b2, c, p = params.values() 

  pprey = y[0:N]  
  vprey = y[N:2*N]  
  ppred = y[2*N:]
  # vpred = y[2*N+1:]  

  # ndarrays of shape (N, N-1, 2)
  dist_prey =  np.array([pprey[i] - np.delete(pprey, i, axis = 0) for i in range(N)])  
  
  vel_prey  =  np.array([vprey[i] - np.delete(vprey, i, axis = 0) for i in range(N)])  

  # perception cone for prey observing prey
  _vprey = vprey[:,np.newaxis,:] + np.zeros_like(dist_prey)
  dot_prod_h = np.sum(_vprey*dist_prey, axis = 2)[:,:,np.newaxis]
  norm_prey = np.linalg.norm(dist_prey, axis = 2)[:,:,np.newaxis]
  abs_val_h  = np.linalg.norm(_vprey, axis=2)[:,:,np.newaxis] * norm_prey
  mask_hh = (dot_prod_h / abs_val_h)>=a_h # (N,N-1,1)
  
  # kernel of prey
  ker_prey = -1 * kh * vel_prey / (1 + (norm_prey**2))
  
  # attraction of prey
  attrac_prey = -1 * a * dist_prey

  # repulsion of prey
  rep_prey = b1 * dist_prey / (norm_prey**2)
                          
  # repulsion of predator (N,2)
  norm_pred = np.linalg.norm(pprey - ppred, axis = 1)[:,np.newaxis]
  rep_pred = b2 * (pprey - ppred) / (norm_pred**2)  
  
  # perception cone for prey observing predator (N,1)
  mask_hp = (np.sum(vprey*(pprey-ppred), axis = 1)[:,np.newaxis] / (norm_pred * np.linalg.norm(vprey)))>=a_h
  
  # acceleration of prey
  acc_prey = (1/N) * np.sum((p2 + mask_hh * (p1-p2)) * (ker_prey + attrac_prey + rep_prey), axis=1) + \
    rep_pred * mask_hp  

  # velocity of predator
  vpred = (1/N) * (c * np.sum((pprey - ppred) / (norm_pred**p), 
                                 axis = 0))#[np.newaxis]  

  # perception cone for predator observing prey 
  # try implementing the perception cone by computed velocity
  dot_prod_p = np.sum((ppred - pprey)*vpred, axis = 1)[:,np.newaxis]
  abs_val_p  = norm_pred * np.linalg.norm(vpred)
  mask_ph    = (dot_prod_p / abs_val_p) >= a_p
  
  # re-computed velocity of predator
  vpred = (1/N) * (c * np.sum(((p2+(p1-p2)*mask_ph)*pprey - ppred) / (norm_pred**p), 
                                 axis = 0))[np.newaxis]
                                 
  return np.concatenate((vprey, acc_prey, vpred))
  
  
  
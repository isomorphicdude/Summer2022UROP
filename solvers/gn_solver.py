"""Implements Gauss-Newton Solver class for nonlinear least squares."""

import numpy as np
from tqdm import tqdm
from numpy.linalg import pinv  


class GaussNewtonSolver(object):

    """
    Gauss-Newton method for gradient descent.  

    - Given true values y
    - Given function f
    - Given inputs x
    - Find optimal parameters theta
    """

    def __init__(self, f, max_solver_iter=100, step=1e-4, 
                    max_line_search_iter=3,
                    verbose = True):

        """
        Initializes the solver.   

        Args:
            - f: function, takes in x of shape (2(2N+1), ) 
                 and theta (the parameter vector),
                 returns y of shape (2n(2N+1), ) as a flattened array 
                 (the 2 comes from x and y coordinates)
            - max_solver_iter: int, maximum iterations
            - step: float, step size for numerical differentiation
            - max_line_search_iter: int, maximum iterations for line search
            - verbose: bool, decides whether to print progress
        """  
        self.max_iter = max_solver_iter
        self.f = f
        self.step = step
        self.max_line= max_line_search_iter
        self.verbose = verbose
        # the attributes to be initialized later
        self.init_guess = None
        self.params = None
        self.x = None
        self.y = None
        self.loss_J = None
        

    def fit(self, x, y, t, init_guess, 
            alpha, beta,
            eps = 1e-4):
        """
        Fits the model.  
        Every time fit is called, x,y are reset.  

        Args:
            - x: array-like, inputs, shape ((2N+1), ) one dimensional
            - y: array-like, true values, shape (2n(2N+1), ) one dimensional
            - t: float, descent step size
            - init_guess: array-like, initial guess for parameters
            - alpha: float, parameter for line search
            - beta: float, parameter for line search
            - eps: float, threshold for algorithm termination
        Returns:
            - self.theta: array-like, optimal parameters
        """  

        self.params = init_guess
        self.x = x
        self.y = y
        
        for i in tqdm(range(self.max_iter)):
            # print  
            if self.verbose:
                print(f"Starting trial {i+1}")
            # compute fitted values  
            y_fitted = self.f(x, self.params)

            # compute residuals
            residual = y_fitted - y

            # compute descent direction
            jacobian = self._jacobian(self.f, self.params, step=self.step)

            dk = self._dK(jacobian)@residual

            # line search
            t = self.line_search(self.params, t, dk, alpha, beta, 
                                    max_iter=self.max_line)

            # update parameters
            self.params = self.params + t * dk

            # if self.verbose:
            print(f"The step size is {t}")
            print(f"The descent direction is {dk}")
            
            # if self.verbose:
            print(f"After trial {i+1}, parameters are {self.params} \n")
            # terminate when gradient < tolerance eps
            if np.linalg.norm(self.loss_J) < eps:
                print("Threshold reached!")
                break
                
        return self.params


    # TODO: implement exact line search also
    def line_search(self, params, t, dk, 
                    alpha, beta, max_iter = 3):
        """
        Performs backtracking line search on the loss function. (see Dante's notes)

        Args:
            - params: array-like, current parameter to be updated
            - t: float, initial step size, should be a bit larger
            - dk: array-like, descent direction  
            - alpha: float between 0 and 1, constant
            - beta: float between 0 and 1, constant
            - max_iter: int, maximum no. of iterations to stop

        Returns:
            - t: float, final step size after searching
        """  

        iter_cnt = 0

        while np.sum((self.f(self.x, params)-self.y)**2) - \
            np.sum((self.f(self.x, params+t*dk))**2)\
            < -1 * alpha * self.loss_jacobian(params, self.step)@dk:

            t = beta * t
            iter_cnt+=1
            if iter_cnt >max_iter:
                print("Max iteration reached for line search!")
                break

        return t


    def loss_jacobian(self, params, step = 1e-4):
        """
        Computes the Jacobian matrix of loss function.  

        Args:  
            - params: array-like, parameters, one dimensional (m, )
            - step: float, step size  

        Returns:
            - J: array-like, Jacobian matrix of loss function  wrt. params
        """       
        # uncomment this line if need to print

        # print("Computing loss jacobian")

        # self.checkValues()
        m = len(params)

        # broadcasting
        params_mat = params[:,np.newaxis] + np.zeros((m, m))

        # creating theta+h and theta-h for finite difference
        params_plus = params_mat + np.eye(m) * step
        params_minus = params_mat - np.eye(m) * step
        y_plus = np.zeros((1, m))
        y_minus = np.zeros((1, m))

        # computing f(theta+h) and f(theta-h)
        for i in range(m):
            y_plus[:,i] = np.sum((self.f(self.x, params_plus[:,i]))**2)
            y_minus[:,i] = np.sum((self.f(self.x, params_minus[:,i]))**2)


        # computing the jacobian
        J = (y_plus - y_minus) / (2 * step)
        # print(J.shape)
        self.loss_J = J
        return J


        

    # can remove the f and params arguments to use self.f, self.params
    def _jacobian(self, f, params, step=1e-4):
        """
        Computes the Jacobian matrix.  

        Args:  
            - f: function to compute Jacobian for
            - params: array-like, parameters, one dimensional (m, )
            - step: float, step size  

        Returns:
            - J: array-like, Jacobian matrix
        """  

        m = len(params)
        M = len(self.y)

        # broadcasting
        params_mat = params[:,np.newaxis] + np.zeros((m, m))



        # creating theta+h and theta-h for finite difference
        params_plus = params_mat + np.eye(m) * step
        params_minus = params_mat - np.eye(m) * step


        # computing f(theta+h) and f(theta-h)
        y_plus = np.zeros((M, m))
        y_minus = np.zeros((M, m))

        # looping over the number of parameters is ok as it's a small number?
        for i in range(m):
            y_plus[:,i] = f(self.x, params_plus[:,i])
            y_minus[:,i] = f(self.x, params_minus[:,i])

        # computing the jacobian
        J = (y_plus - y_minus) / (2 * step)

        # can comment out the following line to improve performance
        # assert J.shape==(M, m), f"J has a shape of {J.shape}"
    

        return J


    @staticmethod
    def jacobian(f, x, params, step=1e-4):
        """
        Computes the Jacobian matrix.  

        Args:  
            - f: function vector valued
            - x: the inputs to the function
            - params: array-like, parameters, one dimensional (m, )
            - step: float, step size  

        Returns:
            - J: array-like, Jacobian matrix
        """  
        m = len(params)

        # broadcasting
        params_mat = params[:,np.newaxis] + np.zeros((m, m))

        # creating theta+h and theta-h for finite difference
        params_plus = params_mat + np.eye(m) * step
        params_minus = params_mat - np.eye(m) * step

        # computing f(theta+h) and f(theta-h)
        y_plus = f(x, params_plus)
        y_minus = f(x, params_minus)


        # computing the jacobian
        J = (y_plus - y_minus) / (2 * step)
        
        return J


    @staticmethod
    def _dK(J):
        """
        Computes the 'descent direction' using pseudo-inverse.  

        Args:
            - J: array-like, Jacobian matrix

        Returns:
            - dk: array-like, (J^T J)^{-1} J^T        
        """  
        # print("Computing dk")

        return pinv(J.T@J)@J.T

    # def checkValues(self):
    #     """Checks if the values are initialized."""
    #     assert self.params is not None \
    #         and self.x is not None\
    #         and self.y is not None, "Must fit model first!"
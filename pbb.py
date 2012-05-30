# Other libraries
import numpy as np
#import pycuda.gpuarray as cua
#import pyfft.cuda as cufft
import scipy.optimize
import os
import time
from PIL import Image

# Own libraries
#import imagetools
#import cnvtools
#import gputools


class MVM_Objective:

    def __init__(self, A, b):
        self.A = A
        self.b = b

    def compute_obj(self, x):

        return 0.5 * np.linalg.norm( np.dot(self.A,x) - self.b )**2
        
    def compute_grad(self, x):

        return np.dot(self.A.T, np.dot(self.A,x) - self.b)

class CNV_Objective:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def compute_obj(self, f):

        f = reshape(f,self.X.sf)
        return 0.5 * np.linalg.norm( self.X.cnv(f) - self.y )**2
        
    def compute_grad(self, f):

        f = reshape(f,self.X.sf)
        return self.X.cnvtp(self.X.cnv(f) - self.y).flatten()


class Solopt:
    """
    SOLOPT  --  Creates a default options structure
     
    OPTIONS = SOLOPT
     
    The fields are described as follows
     
    OPTIONS.MAXIT      -- maximum number of iterations. Default = 150
    OPTIONS.MAXTIME    -- maximum time (in seconds) optimization is allowed to
                          run. In case you do not want this value to impact the 
                          code, set the value of options.time_limit = 0
    OPTIONS.MAXMEM     -- maximum number of vectors for L-BFGS. Default = 5.
    OPTIONS.TIME_LIMIT -- has a default value of 1
    OPTIONS.USE_TOLX and OPTIONS.TOLX -- the first variable determines if
                          relative change in the solution variable 'x' should 
                          be used to determine a stopping criterion. The value 
                          of this tolerance is 1e-6 by default and can be 
                          changed by setting OPTIONS.TOLX
    OPTIONS.USE_TOLO and OPTIONS.TOLO -- stopping criterion based on relative 
                          change in objective function value
    OPTIONS.USE_KKT and OPTIONS.TOLK -- stopping criterion based on KKT
                          violation. Currently only support for nnls problems.
    OPTIONS.VERBOSE    -- whether to print info on screen while optimizing 
                          (default = true)
    OPTIONS.TAU = 1e-5 -- This is a scaled version of the 'tau' parameter 
                          usually smaller values lead to better answers.
    OPTIONS.COMPUTE_OBJ -- whether to compute obj. value at each iteration or 
                          not (default is 1, i.e., yes) set to 0 to save some cpu
    OPTIONS.ASGUI      -- if to report progress of algo by a waitbar (default = 0)
    OPTIONS.PBB_GRADIENT_NORM -- Default is 1e-9
    OPTIONS.MAX_FUNC_EVALS -- maximum number of iterations for
                          line-search. (default = 100).
    OPTIONS.BETA
    and OPTIONS.SIGMA  -- constants for line-search. Defaults are
                          0.0498 and 0.298 respectively.
    OPTIONS.ALGO       -- select the underlying algorithm. Available algorithms 
                          are  
                         'PQN' : projected quasi-Newton,
                         'PLB' : projected limited memory BFGS,
                         Other algos to be added in the future.

    Version 1.0 (c) 2008  Dongmin Kim  and Suvrit Sra
    Translation to python: Michael Hirsch (c) 2011
    """

    def __init__(self):

        self.maxiter = 200;
        self.maxtime = 100;
        self.maxmem  = 17;     # default
        self.maxmem  = 10;     # smaller = faster
        self.time_limit = 0;   
        self.use_tolx = 0;     # who knows!
        self.use_tolo = 1;
        self.use_tolg = 0;
        self.use_kkt  = 0;
        self.tolx     = 1e-8;  # who knows things stop before 10sec!
        self.tolo     = 1e-6;
        self.tolk     = 1e-8;
        self.tolg     = 1e-8;
        self.verbose  = 1;      # initially
        self.tau      = 1e-5;             
        self.compute_obj = 1;
        self.asgui    = 0;
        self.max_func_evals = 10; # might save additional
                                  # time, is used in line search
        self.pbb_gradient_norm = 1e-9;
        self.beta  = 0.0498;
        self.sigma = 0.298;
      
class PBB:
    """
    PBB   --  Optimizes f(x) s.t., x >= 0
     
    This function solves the following optimization problem
        min f(x) subject to x >= 0
     
    The implementation follows a 'reverse-communication' interface wherein
    the function f(x) and its gradient f'(x) are computed via function
    handles.
     
    Usage: 
          OUT = PBB(FX, GFX, x0, OPTIONS)
     
    FX -- function handle to compute f(x)
          it will be invoked as FX(x)
     
      Version 1.0 (c) 2008  Dongmin Kim  and Suvrit Sra
  GFX -- function handle to compute f'(x)
           invoked as GFX(x)
     
    x0 -- Starting vector (useful for warm-starts) -- *should not* be zero.
     
    OPTIONS -- This structure contains important options that control how the
    optimization procedure runs. To obtain a default structure the user can
    use 'options = pbboptions'. Use 'help pbboptions' to get a description of
    what the individual options mean.
     
    OUT contains the solution and other information about the optimization or
    info indicating whether the method succeeded or failed.     
     
    Version 1.0 (c) 2008  Dongmin Kim  and Suvrit Sra
    Translation to python: Michael Hirsch (c) 2011
    """

    def __init__(self, objective, x_init, options):

        self.objective  = objective
        self.options    = options
        self.time_start = time.clock()
        self.iter       = 0
        self.status     = 'Failure'
        
        # ------------------------------------------
        #  Initialisation
        #  -----------------------------------------        
        self.initialisation(x_init)
        
        # ------------------------------------------
        #  Sanitychecks
        #  -----------------------------------------        
        if np.linalg.norm(x_init) < 1e-12:
            raise IOError('Initial vector close to zero. Cannot proceed');

        # ------------------------------------------
        #  Prime the pump
        #  -----------------------------------------
        if options.verbose:
            print 'Running Projected Barzilai Borwein:\n'

        # ------------------------------------------
        #  Main iterative loop
        #  -----------------------------------------        
        for i in arange(options.maxiter):

            self.iter += 1
            self.show_status()        
            
            dx = self.x - self.oldx
            dg = self.g - self.oldg

            dx[(self.x == 0) * (self.g > 0)] = 0
            dg[(self.x == 0) * (self.g > 0)] = 0
            print dx
            print dg
            self.dx = dx
            self.dg = dg

            # Check termination criteria
            self.check_termination()
            if self.term_reason:
                break                

            # store x & gradient
            self.oldx = self.x
            self.oldg = self.g

            # update x & gradient
            if (np.mod(self.iter, 2) == 0):
                step = (dx*dx).sum() / (dx*dg).sum()
            else:
                step = (dx*dg).sum() / (dg*dg).sum()

            self.x = self.x - step * self.g
            self.x[self.x < 0] = 0;      # projection
            self.g = objective.compute_grad(self.x)
    
            if (options.compute_obj):
                self.oldobj = self.obj;
                self.obj = objective.compute_obj(self.x);

        # ------------------------------------------
        #  Final statistics and wrap up
        #  -----------------------------------------        
        self.time   = time.clock() - self.time_start
        self.status = 'Success'

        if self.options.verbose:
            print 'Done\n'

        self.result = self.x

    def initialisation(self, x_init):

        y  = np.random.random(x_init.shape)
        fx = self.objective.compute_obj(x_init)
        fy = self.objective.compute_obj(y)

        if fx < fy:
            self.x      = x_init
            self.oldx   = y
            self.g      = self.objective.compute_grad(x_init)
            self.oldg   = self.objective.compute_grad(y)
            self.obj    = fx
            self.oldobj = fy
            
        else:
            self.x      = y
            self.oldx   = x_init
            self.g      = self.objective.compute_grad(y)
            self.oldg   = self.objective.compute_grad(x_init)
            self.obj    = fy
            self.oldobj = fx    

    def show_status(self):
        
        if self.options.verbose:
            print '.',
            if mod(self.iter , 30) == 0:
                print '\n'
            
    def check_termination(self):
        """
        Check various termination criteria
        """
        
        # First check if we are doing termination based on running time
        if (self.options.time_limit):
            self.time = time.clock - self.time_start
            if (self.time >= self.options.maxtime):
                self.term_reason = 'Exceeded time limit'
                return
         
        # Now check if we are doing break by tolx
        if (self.options.use_tolx):
            if (linalg.norm(self.dx)/linalg.norm(self.oldx) < self.options.tolx):
                self.term_reason = 'Relative change in x small enough'
                return
         
        # Are we doing break by tolo (tol obj val)
        if (self.options.use_tolo and self.iter > 2):
            delta = abs(self.obj-self.oldobj);
            if (delta < self.options.tolo):
                self.term_reason ='Relative change in objvalue small enough'
                return

        # Check if change in x and gradient are small enough 
        if (np.linalg.norm(self.dx) < self.options.tolx) \
               or (np.linalg.norm(self.dg) < self.options.tolg):
            self.term_reason = '|x_t+1 - x_t|=0 or |grad_t+1 - grad_t| < 1e-9'
            return
         
        # Finally the plain old check if max iter has been achieved
        if (self.iter >= self.options.maxiter):
            self.term_reason = 'Maximum number of iterations reached'
            return
         
        # KKT violation
        if (self.options.use_kkt):
            if np.abs(np.sqrt(np.dot(self.x,self.grad))) <= options.tolk:
                self.term_reason = '|x^T * grad| < opt.pbb_gradient_norm'
                return
         
        # Gradient check
        if (self.options.use_tolg):
            nr = linalg.norm(self.grad, inf);
            if (nr < self.options.tolg):
                self.term_reason = '|| grad ||_inf < opt.tolg'
                return
         
        # No condition met, so return false
        self.term_reason = 0;        

if __name__ == '__main__':

    case = 2
    if case == 1:
        A  = pylab.normal(size = (1e5,1e4))
        xt = pylab.rand(1e4,1)
        b = np.dot(A, xt)
         
        x_init = 0.1 * np.ones((1e4,1))
         
        # Set up objective
        objective = MVM_Objective(A,b)
         
        # Default optimization options
        opt = Solopt()
         
        pbb = PBB(objective, x_init, opt); 

    elif case == 2:
        import cnv
        
        x  = pylab.imread('lena.png')
        x  = np.mean(x,2)
        sx = np.shape(x)

        f  = imread('f.png')
        f  = np.mean(f,2)
        f  /= np.sum(f.flatten())
        sf = np.shape(f)

        X = cnv.cnvarray(x, sf, mode = 'valid')
        y = X.cnv(f)

        f_init = np.ones(f.shape)/np.prod(f.shape)
         
        # Set up objective
        objective = CNV_Objective(X,y)
         
        # Default optimization options
        opt = Solopt()
         
        pbb = PBB(objective, f_init.flatten(), opt); 
        fpbb = reshape(pbb.result,sf)

        # For comparison
        flbfgsb = X.deconv(y, maxfun=200)

        figure(1)
        subplot(231)
        imshow(x,'gray')
        title('Sharp image')
        subplot(232)
        imshow(y,'gray')
        title('Blurry image')
        subplot(234)
        imshow(f,'gray')
        title('True image')
        subplot(235)
        imshow(fpbb,'gray')
        title('Estimated f by PBB')              
        subplot(236)
        imshow(flbfgsb[0],'gray')
        title('Estimated f by LBFGSB')

        


        

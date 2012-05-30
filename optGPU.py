# Other libraries
import numpy as np
import pycuda.elementwise as cuelement
import pycuda.curandom as curand
import pycuda.gpuarray as cua
#import pyfft.cuda as cufft
import pylab
import time
from PIL import Image

# Own libraries
#import imagetools
#import cnvtools
import gputools

np.random.seed(120)


class MVM_Objective:

    def __init__(self, A, b):
        self.A = A
        self.b = b

    def compute_obj(self, x):

        return 0.5 * cua.dot( cua.dot(self.A,x) - self.b )
        
    def compute_grad(self, x):

        return cua.dot(self.A.T, cua.dot(self.A,x) - self.b)

class CNV_Objective:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def compute_obj(self, f):

        res_gpu = self.X.cnv(f)-self.y
        return 0.5 * cua.dot(res_gpu,res_gpu)
        
    def compute_grad(self, f):

        return self.X.cnvtp(self.X.cnv(f) - self.y)
    
class Objective(object):
    
    def __init__(self, obj=None, grad=None, both=None):
        self.compute_obj = obj
        self.compute_grad = grad
        self.compute_both = both


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

        self.maxiter = 200
        self.maxtime = 100
        self.maxmem  = 17     # default
        self.maxmem  = 10     # smaller = faster
        self.time_limit = 0   
        self.use_tolx = 0     # who knows!
        self.use_tolo = 1
        self.use_tolg = 0
        self.use_kkt  = 0
        self.tolx     = 1e-8  # who knows things stop before 10sec!
        self.tolo     = 1e-6
        self.tolk     = 1e-8
        self.tolg     = 1e-8
        self.verbose  = 0;      # initially
        self.tau      = 1e-5             
        self.compute_obj = 1
        self.compute_both = 0
        self.asgui    = 0
        self.max_func_evals = 10 # might save additional
                                 # time, is used in line search
        self.pbb_gradient_norm = 1e-9
        self.beta  = 0.0498
        self.sigma = 0.298
        self.unconstrained = False
        
clip2bound = cuelement.ElementwiseKernel(
    "float *dx, float *x, float *g",
    "dx[i] = ((x[i] == 0.f) && (g[i] > 0)) ? 0.f : dx[i]",
    "clip2bound")
      
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
        #  Sanity checks
        #  -----------------------------------------
        if np.sqrt(cua.dot(self.x, self.x).get()) < 1e-12:
            raise IOError('Initial vector close to zero. Cannot proceed');

        # ------------------------------------------
        #  Prime the pump
        #  -----------------------------------------
        if options.verbose:
            print 'Running Projected Barzilai Borwein:\n'

 
        # ------------------------------------------
        #  Main iterative loop
        #  -----------------------------------------        
        for i in range(options.maxiter):
            self.iter += 1
            self.show_status()        
            
            dx = self.x - self.oldx
            dg = self.g - self.oldg

            if not options.unconstrained:
                clip2bound(dx, self.x, self.g)
                clip2bound(dg, self.x, self.g)                
    
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
                step = (cua.sum(dx*dx) / (0.00001+cua.sum(dx*dg))).get()
            else:
                step = (cua.sum(dx*dg) / (0.00001+cua.sum(dg*dg))).get()
        
            self.x = self.x - self.g * step
            if not options.unconstrained:
                gputools.cliplower_GPU(self.x, 0)      # projection
            
    
            if options.compute_both:
                self.oldobj = self.obj
                self.obj, self.g = objective.compute_both(self.x);
            elif options.compute_obj:
                self.g = objective.compute_grad(self.x)
                self.oldobj = self.obj;
                self.obj = objective.compute_obj(self.x);
            else:
                self.g = objective.compute_grad(self.x)
                

        # ------------------------------------------
        #  Final statistics and wrap up
        #  -----------------------------------------        
        self.time   = time.clock() - self.time_start
        self.status = 'Success'

        if self.options.verbose:
            print self.status
            print self.term_reason
            print 'Done\n'

        self.result = self.x

    def initialisation(self, x_init):

        y  = curand.rand(x_init.shape)
        y  -= x_init + 0.5 # this is only a fixx, remove this line if possible
        if self.options.compute_both:
            fx, gx = self.objective.compute_both(x_init)
            fy, gy = self.objective.compute_both(y)
            
            if fx < fy:
                self.x      = x_init
                self.oldx   = y
                self.g      = gx
                self.oldg   = gy
                self.obj    = fx
                self.oldobj = fy
              
            else:
                self.x      = y
                self.oldx   = x_init
                self.g      = gy
                self.oldg   = gx
                self.obj    = fy
                self.oldobj = fx
        else:
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
            if np.mod(self.iter , 30) == 0:
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
            if (np.sqrt(cua.dot(self.dx,self.dx).get())/
                np.sqrt(cua.dot(self.oldx,self.oldx).get()) < self.options.tolx):
                self.term_reason = 'Relative change in x small enough'
                return
         
        # Are we doing break by tolo (tol obj val)
        if (self.options.use_tolo and self.iter > 2):
            delta = abs(self.obj-self.oldobj)
            if (delta < self.options.tolo):
                self.term_reason ='Relative change in objvalue small enough'
                return

        # Check if change in x and gradient are small enough
        # we don't want that for now
#        if (np.sqrt((cua.dot(self.dx,self.dx).get())) < self.options.tolx) \
#               or (np.sqrt(cua.dot(self.dg,self.dg).get()) < self.options.tolg):
#            self.term_reason = '|x_t+1 - x_t|=0 or |grad_t+1 - grad_t| < 1e-9'
#            return
         
        # Finally the plain old check if max iter has been achieved
        if (self.iter >= self.options.maxiter):
            self.term_reason = 'Maximum number of iterations reached'
            return
         
        # KKT violation
        if (self.options.use_kkt):
            if np.abs(np.sqrt(cua.dot(self.x,self.grad).get())) <= options.tolk:
                self.term_reason = '|x^T * grad| < opt.pbb_gradient_norm'
                return
         
        # Gradient check
        if (self.options.use_tolg):
            nr = cua.max(cua.fabs(self.grad)).get();
            if (nr < self.options.tolg):
                self.term_reason = '|| grad ||_inf < opt.tolg'
                return
         
        # No condition met, so return false
        self.term_reason = 0;        

if __name__ == '__main__':

    case = 2
    if case == 1:
        A  = curand.rand((10000,1000))
        xt = curand.rand((1000,1))
        b  = cua.dot(A, xt)
         
        x_init = cua.empty_like(xt)
        x_init.fill(0.1)
         
        # Set up objective
        objective = MVM_Objective(A,b)
         
        # Default optimization options
        opt = Solopt()
         
        pbb = PBB(objective, x_init, opt); 

    elif case == 2:
        
        x  = pylab.imread('lena.png')
        x  = np.mean(x,2)
        sx = np.shape(x)

        f  = pylab.imread('f.png')
        f  = np.mean(f,2)
        f  /= np.sum(f.flatten())
        sf = np.shape(f)

        import cnvGPU as cnv
        X = cnv.CnvGPU(x, sf, mode='valid')
        y = X.cnv(f)

        f_init = cua.empty(f.shape, np.float32)
        f_init.fill(1./np.prod(f.shape))
         
        # Set up objective
        objective = CNV_Objective(X,y)
         
        # Default optimization options
        opt = Solopt()
         
        pbb = PBB(objective, f_init, opt); 

        # For comparison
        flbfgsb = X.deconv(y, maxfun=200)

        pylab.figure(1)
        pylab.subplot(231)
        pylab.imshow(x,'gray')
        pylab.title('Sharp image')
        pylab.subplot(232)
        pylab.imshow(y.get(),'gray')
        pylab.title('Blurry image')
        pylab.subplot(234)
        pylab.imshow(f,'gray')
        pylab.title('True image')
        pylab.subplot(235)
        pylab.imshow(pbb.result.get(),'gray')
        pylab.title('Estimated f by PBB')              
        pylab.subplot(236)
        pylab.imshow(flbfgsb[0],'gray')
        pylab.title('Estimated f by LBFGSB')
        pylab.show()

        


        

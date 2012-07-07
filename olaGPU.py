#!/usr/bin/env python
# Other libraries
import pycuda.gpuarray as cua
import pyfft.cuda as cufft
import scipy.misc
import numpy as np

# Own libraries
import gputools

class OlaGPU:
    """
    General description:
    
    OlaGPU creates a spatially-varying convolution matrix for 2-dimensional
    numpy arrays. It's a 2d generalization of the overlap-and-add short-time
    Fourier transform. It allows the computation of the forward convolution
    as well as its transpose operation, i.e. by denoting the forward model as

        y = Xf = Fx

    where y is the blurry image, f the spatially-varying PSF and x the sharp
    image, this class allows the computation of Xf, Fx as well as X'y and 
    F'y. All of these four operations are needed for blind deconvolution
    featuring gradient based optimization.

    Note, that this class caches the FFT of the initialising object, i.e.
    either the image in case of creating an instance of X or the PSF in case
    of creating an instance of F. This class implements also some basic
    deconvolution algorithms.
    --------------------------------------------------------------------------
    Usage:
    
    Call:  Z = OlaGPU(z, sw, mode, winaux)
    
    Input: z        either PSF or image
           sw       shape of array the convolution matrix is applied to
           mode     a flag indicating the size of the output
                    'valid'  (0): The output consists only of those elements
                                  that do not rely on the zero-padding, i.e.
                                  sy = sx - sf + 1
                                  
                    'same'   (1): The output is the same size as the input
                                  centered with respect to the 'full' output
                                  sy = sx
                                  
                    'full'   (2): The output is the full discrete linear
                                  convolution of the inputs. 
                                  sy = sx + sf - 1
                                  Not fully tested.

                    'circ'   (3): The output is the same size as the input
                                  centered with respect to the 'full' output
                                  sy = sx
                                  The difference to (1) is that no zero-
                                  padding has been applied, i.e. circular
                                  boundary conditions are assumed.
                                  Not fully tested.

           winaux   instance of win2winaux class containing the windows
                    which determine the interpolation scheme between
                    neighboring PSF kernels. See win2winaux for more info.

    Output: Class object which provides the following methods:

           cnv          applies convolution matrix to an array of size sw 
           cnvtp        computes transpose operation, i.e. correlation 
           devonv       performs deconvolution given some blurry input image
           devonv_rgb   same as deconv for color images

    See the description of the individual methods for more details and info.

    --------------------------------------------------------------------------
    Dependencies:

    This library requires PyCuda (tested with version 2011.1.2 and
    version 0.94.2) as well as the pyfft package (tested with version 0.3.5).
    See Andreas Kloeckner's project homepage
    
    http://mathema.tician.de/software/pycuda
    
    for details on PyCuda and its dependencies as well as

    http://pypi.python.org/pypi/pyfft.

    for the installation guide of pyfft.
    --------------------------------------------------------------------------
    Contact:

    michael.hirsch@ucl.ac.uk

    Copyright (C) 2011 Michael Hirsch
    """

    def __init__(self, f, sx, mode, winaux):
            
        sf     = np.array(f.shape)[-2::]
        sx     = np.array(sx)
        sw     = winaux.sw
        csf    = winaux.csf
        nhop   = winaux.nhop
    
        # Check what is f and what is x
        if (len(f.shape) == 3) and all(sx > sf):
            self.f = f
            self.x = []
            self.__id__ = 'F'

            # Safety check
            if np.prod(f.shape[0]) != np.prod(csf):
                raise IOError('Size missmatch between winaux and PSF size!')
        
        elif (len(f.shape) == 2) and all(sx < sf):
            self.f = []
            self.x = f
            self.__id__ = 'X'
            sf = sx
            sx = np.array(self.x.shape)
            
        elif any(sf < sx) and any(sf > sx):
            raise IOError('Size missmatch')

        # Safety check
        if any(winaux.sx != sx):
            raise IOError('Size missmatch between winaux and image size!')

        if mode == 'valid':
            sy   = sx - sf + 1
        elif mode == 'same':
            sy   = sx
        elif mode == 'full':
            sy   = sx + sf - 1
        elif mode == 'circ':
            sy   = sx
        else:
            raise NotImplementedError('Not a valid mode!')      

        # Pad either f or x to be sized a power of 2 and copy it to device
        sfft     = sw + sf - 1
        sfft_gpu = (2**np.ceil(np.log2(sfft)))
        sfft_gpu = (int(sfft_gpu[0]),int(sfft_gpu[1]))
        if self.__id__ == 'F':
            # each kernel of PSF has to be padded
            fft_gpu = gputools.pad_stack_GPU(self.f, sfft_gpu, dtype='complex')
            self.sz = sx
            
        elif self.__id__ == 'X':
            # each patch has to be modulated by window
            fft_gpu = gputools.chop_mod_pad_GPU(self.x, winaux.ws_gpu, csf,
                                                sw, nhop, sz=sfft_gpu,
                                                dtype='complex')    
            self.sz = sf

        # Create FFT plan and compute FFT
        plan = cufft.Plan(fft_gpu.shape[-2::])
        self.plan = plan
        self.fft(fft_gpu, fft_gpu.shape[0])
        
        self.sfft_gpu = sfft_gpu
        self.fft_gpu  = fft_gpu
        self.winaux   = winaux
        self.sfft = sfft
        self.sf   = sf
        self.sx   = sx
        self.sy   = sy
        self.mode = mode



    def cnv(self, u):
        """
        Description:
        
        cnv computes the correlation of the convolution matrix with either
        an image or PSF whether the parent class is an instance of F or X,
        i.e. Fx or Xf respectively.
        ----------------------------------------------------------------------
        Usage:
    
        Call:  Z = OlaGPU(z, sw, mode, winaux)
               y = Z.cnv(u)

        Input:  u   either image of PSF 
        Ouput:  y   a blurry image       
        """

        # Pad either f or x and copy it to device
        if (len(u.shape) == 3) and (self.__id__ == 'X'):
            # Safety  check
            if np.prod(u.shape[0]) != np.prod(self.winaux.csf):
                raise IOError('Size missmatch between winaux and PSF size!')
            u_gpu =  gputools.pad_stack_GPU(u, self.sfft_gpu, dtype='complex')

        elif (len(u.shape) == 2) and (self.__id__ == 'F'):
            # Safety check
            if sum(u.shape != self.winaux.sx) > 0:
                raise IOError('Size missmatch between winaux and image size!')

            # Chop input image into overlapping patches, modulate them
            # by windows and do appropriate padding            
            #u_gpu = gputools.chop_mod_pad_GPU(u, self.winaux.ws_gpu,
            #                              self.winaux.csf, self.winaux.sw,
            #                              self.winaux.nhop, sz=self.sfft_gpu,
            #                              dtype='complex')   

            ############
            # Something is wrong in above kernel, which should perform
            # modulation and padding in one kernel call. For now workaround:
            offset = (0,0)
            u_gpu = gputools.chop_pad_GPU(u, self.winaux.csf,
                                         self.winaux.sw, self.winaux.nhop,
                                         self.sfft_gpu, offset, dtype='complex')

            ws_gpu = gputools.pad_stack_GPU(self.winaux.ws_gpu, self.sfft_gpu,
                                            offset, dtype='complex')
            self.ws = ws_gpu
            u_gpu = ws_gpu * u_gpu
            # Workauround ends here
            ############

        # Compute FFT of input, do multiplication in Fourier space
        # and compute inverse Fourier transform
        self.fft(u_gpu, self.fft_gpu.shape[0])
        
        # Strange enough: inverse does not work due to some error in pyfft
        # Therefore compute the inverse via conj(F(conj(x)))/length(x)
        # see Wikipedia for reference
        ys_gpu = (self.fft_gpu * u_gpu).conj()
        del u_gpu
        self.fft(ys_gpu, self.fft_gpu.shape[0])
        ys_gpu = ys_gpu.conj()/np.prod(ys_gpu.shape[-2::])

        # Do overlap and add
        y_gpu = gputools.ola_GPU_test(ys_gpu, self.winaux.csf,
                                      self.sf-1+self.winaux.sw,
                                      self.winaux.nhop)

        # Do cropping to correct output size
        if self.mode == 'valid':
            y = gputools.crop_gpu2cpu(y_gpu, self.sy, self.sf-1)
        elif self.mode == 'same':
            y = gputools.crop_gpu2cpu(y_gpu, self.sy, np.floor(self.sf/2))
        elif self.mode == 'full':
            y = gputools.crop_gpu2cpu(y_gpu, self.sy)            
        elif self.mode == 'circ':
            if u.__class__ == cua.GPUArray:
                raise NotImplementedError('Not a valid mode!')
            else:
                y = np.real(y_gpu.get())
                y = imagetools.circshift(y, floor(self.sf/2))
        else:
            raise NotImplementedError('Not a valid mode!')

        if u.__class__ == np.ndarray:
            return np.array(y.get())
        elif u.__class__ == cua.GPUArray:
            return y



    def cnvtp(self, y):
        """
        cnvtp computes the correlation of the convolution matrix with
        either an image or a PSF whether , i.e. X'y or F'y respectively.
        ----------------------------------------------------------------------
        Usage:
    
        Call:  Z = OlaGPU(z, sw, mode, winaux)
               u = Z.cnvtp(y)
 
        Input:  y   blurry image
        Ouput:  u   either image or PSF sized object
        """

        # Do chopping and padding of input
        if self.mode == 'valid':
            y_gpu = gputools.chop_pad_GPU(y, self.winaux.csf,
                                        self.winaux.sw, self.winaux.nhop,
                                        self.sfft_gpu, self.sf-1,'complex')
        elif self.mode == 'same':
            y_gpu = gputools.chop_pad_GPU(y, self.winaux.csf,
                                        self.winaux.sw, self.winaux.nhop,
                                        self.sfft_gpu, np.floor(self.sf/2),
                                        'complex')
        else:
            raise NotImplementedError('Not a valid mode!')

        # Compute FFT
        self.fft(y_gpu, self.fft_gpu.shape[0])

        z_gpu = cua.empty_like(y_gpu)

        # Computing the inverse FFT
        # z_gpu = (y_gpu * self.fft_gpu.conj()).conj()
        z_gpu = y_gpu.conj() * self.fft_gpu
        self.fft(z_gpu, self.fft_gpu.shape[0])
        z_gpu = z_gpu.conj()/np.prod(z_gpu.shape[-2::])

        # Do cropping to correct output size
        if self.__id__ == 'X':
            zc_gpu = gputools.crop_stack_GPU(z_gpu, self.sz)
            return zc_gpu
       
        elif self.__id__ == 'F':
            zs_gpu = gputools.crop_stack_GPU(z_gpu, self.winaux.sw)
            zs_gpu = self.winaux.ws_gpu * zs_gpu
            zc_gpu = gputools.ola_GPU_test(zs_gpu, self.winaux.csf,
                                           self.winaux.sw+self.sf-1,
                                           self.winaux.nhop)
            zc_gpu = gputools.crop_gpu2cpu(zc_gpu, self.sx)                 
            return zc_gpu


    
    def deconv(self, y, z0=None, mode='lbfgsb', maxfun=100, alpha=0., beta=0.01,
               verbose=10, m=5, edgetapering=1, factor=3, gamma=1e-4):
        """
        deconv implements various deconvolution methods. It expects a
        blurry image and outputs an estimated blur kernel or a sharp latent
        image. Currently, the following algorithms are implemented:

        'lbfgsb'   uses the lbfgsb optimization code to minimize the following
                   constrained regularized problem:

                   |y-Zu|^2 + alpha * |grad(u)|^2 + beta * |u|^2 s.t. u>0

                   The alpha term promotes smoothness of the solution, while
                   the beta term is an ordinary Thikhonov regularization 
                   
        'direct'   as above but solves the problem directly, i.e. via
                   division in Fourier space instead of an iterative
                   minimization scheme at the cost of the positivity
                   constraint.

        'xdirect'  as 'direct' but without corrective term which reduces
                   artifacts stemming from the windowing

        'gdirect'  solves the following problem

                   |grad(y)-grad(Zu)|^2 + alpha * |grad(u)|^2 + beta * |u|^2

                   This is particularly useful for kernel estimation in the
                   case of blurred natural images featuring many edges. The
                   advantage vs. 'direct' is the suppression of noise in the
                   estimated PSF kernels. 

        'xdirect'  as 'direct' but without corrective term which reduces
                   artifacts stemming from the windowing

                   'Fast Image Deconvolution using Hyper-Laplacian Priors'
                   by Dilip Krishnan and Rob Fergus, NIPS 2009.
                   It minimizes the following problem

                   |y-Zu|^2 + gamma * |grad(u)|^(2/3)

                   via half-quadratic splitting. See paper for details.
                   
        ----------------------------------------------------------------------
        Usage:
    
        Call:  Z = OlaGPU(z, sw, mode, winaux)
               u = Z.deconv(y)
 
        Input:  y   blurry image
        Ouput:  u   either image or PSF sized object
        """

        from numpy import array
        
        if not all(array(y.shape) == self.sy):
            raise IOError ('Sizes incompatible. Expected blurred image!')

        # Potential data transfer to GPU
        if y.__class__ == cua.GPUArray:
            y_gpu = 1. * y
        else:
            y_gpu = cua.to_gpu(y.astype(np.float32))


        # --------------------------------------------------------------------
        if mode == 'lbfgsb':

            from scipy.optimize import fmin_l_bfgs_b
                
            self.res_gpu = cua.empty_like(y_gpu)

            if self.__id__ == 'X':
                sz = ((int(np.prod(self.winaux.csf)),
                       int(self.sz[0]),int(self.sz[1])))
                
            elif self.__id__ == 'F':
                sz = self.sz            
                
            lf = np.prod(sz)
            if z0 == None:
                z0_gpu = self.cnvtp(y_gpu)
                z0 = z0_gpu.get()
                z0 = z0.flatten()
                
                #z0 = np.ones(lf)/(1. * lf)    # initialisation with flat kernels
            else:
                z0 = z0.flatten()

            lb = 0.                 # lower bound
            ub = np.infty           # upper bound
            zhat = fmin_l_bfgs_b(func = self.cnvinv_objfun, x0 = z0, \
                                 fprime = self.cnvinv_gradfun,\
                                 args = [sz, y_gpu, alpha, beta],\
                                 factr = 10., pgtol = 10e-15, \
                                 maxfun = maxfun, bounds = [(lb, ub)] * lf,\
                                 m = m, iprint = verbose)

            return np.reshape(zhat[0], sz), zhat[1], zhat[2]        


        # --------------------------------------------------------------------
        elif mode == 'gdirect':
            
            # Use this method only for estimating the PSF
            if self.__id__ != 'X':
                raise Exception('Use direct mode for image estimation!')
    
            # Compute Laplacian
            if alpha > 0.:
                gx_gpu = gputools.pad_cpu2gpu(
                    np.array([[-1,1],[-1,1],[-1,1]]),
                    self.sfft_gpu, dtype='complex')
                
                gy_gpu = gputools.pad_cpu2gpu(
                    np.array([[-1,-1,-1],[1,1,1]]),
                    self.sfft_gpu, dtype='complex')
                
                self.plan.execute(gx_gpu)
                self.plan.execute(gy_gpu)
                L_gpu = gx_gpu * gx_gpu.conj() + gy_gpu * gy_gpu.conj()
            else:
                L_gpu = cua.zeros(self.fft_gpu.shape, np.complex64)
                 
            if edgetapering == 1:
                gputools.edgetaper_gpu(y_gpu, 2*self.sf, 'barthann')

            # Transfer to GPU
            if self.x.__class__ == cua.GPUArray:
                x_gpu = self.x
            else:
                x_gpu = cua.to_gpu(self.x)        

            # Compute gradient images             
            xx_gpu, xy_gpu = gputools.gradient_gpu(x_gpu)
            yx_gpu, yy_gpu = gputools.gradient_gpu(y_gpu)

            # Chop and pad business
            if self.mode == 'valid':
                yx_gpu = gputools.chop_pad_GPU(yx_gpu, self.winaux.csf,
                                               self.winaux.sw, self.winaux.nhop,
                                               self.sfft_gpu, self.sf-1,
                                               'complex')                
                yy_gpu = gputools.chop_pad_GPU(yy_gpu, self.winaux.csf,
                                               self.winaux.sw, self.winaux.nhop,
                                               self.sfft_gpu, self.sf-1,
                                               'complex')
                
            elif self.mode == 'same':
                yx_gpu = gputools.chop_pad_GPU(yx_gpu, self.winaux.csf,
                                               self.winaux.sw, self.winaux.nhop,
                                               self.sfft_gpu,
                                               np.floor(self.sf/2), 'complex')
                yy_gpu = gputools.chop_pad_GPU(yy_gpu, self.winaux.csf,
                                               self.winaux.sw, self.winaux.nhop,
                                               self.sfft_gpu,
                                               np.floor(self.sf/2), 'complex')
            else:
                raise NotImplementedError('Not a valid mode!')

            xx_gpu = gputools.chop_pad_GPU(xx_gpu, self.winaux.csf,
                                           self.winaux.sw, self.winaux.nhop,
                                           self.sfft_gpu, dtype='complex')
            xy_gpu = gputools.chop_pad_GPU(xy_gpu, self.winaux.csf,
                                           self.winaux.sw, self.winaux.nhop,
                                           self.sfft_gpu, dtype='complex')            

            # Here each patch should be windowed to reduce ringing artifacts,
            # however since we are working in the gradient domain, the effect
            # is negligible
            # ws_gpu = gputools.pad_stack_GPU(self.winaux.ws_gpu,
            #                                 self.sfft_gpu, self.sf-1,
            #                                 dtype='complex') 
            # xx_gpu = ws_gpu * xx_gpu
            # xy_gpu = ws_gpu * xy_gpu
            # yx_gpu = ws_gpu * yx_gpu
            # yy_gpu = ws_gpu * yy_gpu

            # Compute Fourier transform
            self.fft(yx_gpu, self.fft_gpu.shape[0])
            self.fft(yy_gpu, self.fft_gpu.shape[0])
            self.fft(xx_gpu, self.fft_gpu.shape[0])
            self.fft(xy_gpu, self.fft_gpu.shape[0])

            # Do division in Fourier space
            z_gpu = cua.zeros(xy_gpu.shape, np.complex64)            
            z_gpu = gputools.comp_ola_gdeconv(xx_gpu, xy_gpu,
                                              yx_gpu, yy_gpu,
                                              L_gpu, alpha, beta)

            # Computing the inverse FFT
            z_gpu = z_gpu.conj() 
            self.fft(z_gpu, self.fft_gpu.shape[0])
            z_gpu = z_gpu.conj()/np.prod(z_gpu.shape[-2::])

            # Crop out the kernels
            zc_gpu = gputools.crop_stack_GPU(z_gpu, self.sf)
            return zc_gpu
 

        # --------------------------------------------------------------------
        elif mode == 'direct':

            const_gpu = cua.empty_like(y_gpu)
            const_gpu.fill(1.)

            # First deconvolution without corrective term
            y_gpu = self.deconv(y_gpu, mode = 'xdirect', alpha = alpha,
                                beta = beta, edgetapering = edgetapering)      
            gputools.cliplower_GPU(y_gpu,0)

            # Now same for constant image to get rid of window artifacts
            if edgetapering == 1:
                gputools.edgetaper_gpu(const_gpu, 2*self.sf, 'barthann')
                
            const_gpu = self.deconv(const_gpu, mode = 'xdirect', alpha = alpha,
                                    beta = beta, edgetapering = edgetapering)
            gputools.edgetaper_gpu(const_gpu, 2*self.sf, 'barthann')
            gputools.clip_GPU(const_gpu, 0.01, 10.)            

            # Division of deconvolved latent and constant image to get rid
            # of artifacts stemming from windowing
            y_gpu = y_gpu / const_gpu
            sz    = y_gpu.shape
            #gputools.clip_GPU(y_gpu, 0., 1.0)
            #gputools.edgetaper_gpu(y_gpu, 3*self.sf, 'barthann')

            # Do cropping and padding since edges are corrupted by division
            y_gpu = gputools.crop_gpu2cpu(y_gpu, sz-factor*self.sf-1,
                                        offset=np.floor((factor*self.sf-1)/2.))
            y_gpu = gputools.impad_gpu(y_gpu, tuple(np.array(sz)-y_gpu.shape))

            return y_gpu


        # --------------------------------------------------------------------
        elif mode == 'xdirect':

            # Compute Laplacian
            if alpha > 0.:
                gx_gpu = gputools.pad_cpu2gpu(
                    np.array([[-1,1]]), self.sfft_gpu, dtype='complex')
                gy_gpu = gputools.pad_cpu2gpu(
                    np.array([[-1],[1]]), self.sfft_gpu, dtype='complex')
                self.plan.execute(gx_gpu)
                self.plan.execute(gy_gpu)
                L_gpu = gx_gpu * gx_gpu.conj() + gy_gpu * gy_gpu.conj()
            else:
                L_gpu = cua.zeros(self.fft_gpu.shape, np.complex64)

            # Edgetapering of blurry input image
            if edgetapering == 1:
                gputools.edgetaper_gpu(y_gpu, 3*self.sf, 'barthann')
                
            if self.mode == 'valid':
                #y_gpu = gputools.pad_cpu2gpu(y_gpu, self.sx, self.sf-1, dtype='real')
                offset = self.sf-1
            elif self.mode == 'same':
                offset = np.floor(self.sf/2)
            else:
                raise NotImplementedError('Not a valid mode!')

            # Chop and pad business
            y_gpu = gputools.chop_pad_GPU(y, self.winaux.csf,
                                          self.winaux.sw, self.winaux.nhop,
                                          self.sfft_gpu, offset, 'complex')     
            ws_gpu = gputools.pad_stack_GPU(self.winaux.ws_gpu, self.sfft_gpu,
                                            dtype='complex')

            # Windowing
            y_gpu  = ws_gpu * y_gpu

            # Compute FFT
            self.fft(y_gpu, self.fft_gpu.shape[0])

            # Do division in Fourier space
            z_gpu = gputools.comp_ola_deconv(self.fft_gpu, y_gpu, L_gpu,
                                             alpha, beta)

            # Computing the inverse FFT
            z_gpu = z_gpu.conj() 
            self.fft(z_gpu, self.fft_gpu.shape[0])
            z_gpu = z_gpu.conj()/np.prod(z_gpu.shape[-2::])

            # Crop the solution to correct output size 
            if self.__id__ == 'X':
                zc_gpu = gputools.crop_stack_GPU(z_gpu, self.sf)                
                return zc_gpu
       
            elif self.__id__ == 'F':
                zs_gpu = gputools.crop_stack_GPU(z_gpu, self.winaux.sw)
                #zs_gpu = self.winaux.ws_gpu * zs_gpu
                zc_gpu = gputools.ola_GPU_test(zs_gpu, self.winaux.csf,
                                           self.winaux.sw, self.winaux.nhop)
                zc_gpu = gputools.crop_gpu2cpu(zc_gpu, self.sx)             
                return zc_gpu


        # --------------------------------------------------------------------
        elif mode == 'sparse':

            # Compute Laplacian
            gx_gpu = gputools.pad_cpu2gpu(np.sqrt(2.)/2. *
                np.array([[-1,1]]), self.sfft_gpu, dtype='complex')
            gy_gpu = gputools.pad_cpu2gpu(np.sqrt(2.)/2. *
                np.array([[-1],[1]]), self.sfft_gpu, dtype='complex')
            self.plan.execute(gx_gpu)
            self.plan.execute(gy_gpu)
            L_gpu = gx_gpu * gx_gpu.conj() + gy_gpu * gy_gpu.conj()

            const_gpu = cua.empty_like(y_gpu)
            const_gpu.fill(1.)

            # Edgetapering
            if edgetapering == 1:
                gputools.edgetaper_gpu(y_gpu, 2*self.sf, 'barthann')
                gputools.edgetaper_gpu(const_gpu, 2*self.sf, 'barthann')

            # Parameter settings
            beta = 1.
            beta_rate = 2. * np.sqrt(2.)
            beta_max = 2.**8

            # Initialisation of x with padded version of y
            x_gpu = 1 * y_gpu
            if self.mode == 'valid':
                offset = self.sf-1
            elif self.mode == 'same':
                offset = np.floor(self.sf/2)
            else:
                raise NotImplementedError('Not a valid mode!')

            # Chop and pad business
            y_gpu = gputools.chop_pad_GPU(y_gpu, self.winaux.csf,
                                          self.winaux.sw, self.winaux.nhop,
                                          self.sfft_gpu, offset,'complex')
            const_gpu = gputools.chop_pad_GPU(const_gpu, self.winaux.csf,
                                          self.winaux.sw, self.winaux.nhop,
                                          self.sfft_gpu, offset,'complex')
            ws_gpu = gputools.pad_stack_GPU(self.winaux.ws_gpu, self.sfft_gpu,
                                            offset, dtype='complex')

            # Windowing
            y_gpu = y_gpu * ws_gpu

            # Constant image for corrective weighting term
            const_gpu = const_gpu * ws_gpu
            del ws_gpu
            
            self.fft(const_gpu, self.fft_gpu.shape[0])
            const_gpu = gputools.comp_ola_deconv(self.fft_gpu, const_gpu,
                                                 L_gpu, alpha, gamma)
            const_gpu = const_gpu.conj()
            self.fft(const_gpu, self.fft_gpu.shape[0])
            const_gpu = const_gpu.conj()/np.prod(const_gpu.shape[-2::])
            const_gpu = gputools.crop_stack_GPU(const_gpu, self.winaux.sw)
            const_gpu = const_gpu * self.winaux.ws_gpu
            const_gpu = gputools.ola_GPU_test(const_gpu, self.winaux.csf,
                                              self.winaux.sw, self.winaux.nhop)
            const_gpu = gputools.crop_gpu2cpu(const_gpu, self.sx)
            # For debugging purposes
            #scipy.misc.imsave('const1.png', const_gpu.get()/const_gpu.get().max())
            gputools.cliplower_GPU(const_gpu, 0.01)
            const_gpu = 0.01 / const_gpu
            
            # Precompute F'y
            self.fft(y_gpu, self.fft_gpu.shape[0])
            y_gpu = y_gpu * self.fft_gpu.conj()
            
            
            while beta < beta_max:
                # Compute gradient images of x
                xx_gpu, xy_gpu = gputools.gradient_gpu(x_gpu)
                del x_gpu

                # w sub-problem for alpha 2/3
                gputools.modify_sparse23_gpu(xx_gpu, beta) 
                gputools.modify_sparse23_gpu(xy_gpu, beta)
                #gputools.modify_sparse_gpu(xx_gpu, beta, 0.01)
                #gputools.modify_sparse_gpu(xy_gpu, beta, 0.01)

                # Chop and pad to size of FFT
                xx_gpu = gputools.chop_pad_GPU(xx_gpu, self.winaux.csf,
                                           self.winaux.sw, self.winaux.nhop,
                                           self.sfft_gpu, dtype='complex')
                xy_gpu = gputools.chop_pad_GPU(xy_gpu, self.winaux.csf,
                                           self.winaux.sw, self.winaux.nhop,
                                           self.sfft_gpu, dtype='complex')  

                # Compute Fourier transform
                self.fft(xx_gpu, self.fft_gpu.shape[0])
                self.fft(xy_gpu, self.fft_gpu.shape[0])

                # Do division in Fourier space
                x_gpu = gputools.comp_ola_sdeconv(gx_gpu, gy_gpu,
                                                  xx_gpu, xy_gpu,
                                                  y_gpu, self.fft_gpu,
                                                  L_gpu, alpha, beta, gamma)
                del xx_gpu, xy_gpu

                # Computing the inverse FFT
                x_gpu = x_gpu.conj()
                self.fft(x_gpu, self.fft_gpu.shape[0])
                x_gpu = x_gpu.conj()
                x_gpu /= np.prod(x_gpu.shape[-2::])

                # Ola and cropping
                x_gpu = gputools.crop_stack_GPU(x_gpu, self.winaux.sw)
                x_gpu = x_gpu * self.winaux.ws_gpu
                x_gpu = gputools.ola_GPU_test(x_gpu, self.winaux.csf,
                                           self.winaux.sw, self.winaux.nhop)
                x_gpu = gputools.crop_gpu2cpu(x_gpu, self.sx)

                # Enforce positivity
                x_gpu = x_gpu * const_gpu
                gputools.cliplower_GPU(x_gpu, 0.)

                beta *= beta_rate

            return x_gpu
                       
        else:
            raise NotImplementedError('Not a valid deconv mode!')   


    def deconv_rgb(self, y, z0 = None, mode = 'lbfgsb', maxfun = 100,
                   iprint = 10, m = 5, alpha = 0., beta = 0.01,
                   edgetapering = 1):

        """
        Same as deconv for rgb images. Does deconvolution on each color
        channel separately.
        """
        
        y_result = np.empty((self.sx[0],self.sx[1],3), dtype=np.float32)
        for i in range(3):
            y_temp = y[...,i].astype(np.float32).copy()
            y_result[...,i] = self.deconv(y_temp, z0, mode, maxfun, iprint,
                                          m, alpha, beta, edgetapering).get()
        return y_result
    

    def cnvinv_objfun(self, z, sz, y_gpu, alpha=0., beta=0.):
        """
        Computes objective function value of 'lbfgsb' mode of deconv method.
        See deconv for details.
        """
         
        if z.__class__ == np.ndarray:
            z = np.array(np.reshape(z,sz)).astype(np.float32)
            z_gpu = cua.to_gpu(z)            
                
        self.res_gpu = y_gpu - self.cnv(z_gpu)        
 
        obj = 0.5*(cua.dot(self.res_gpu,self.res_gpu,dtype=np.float64))

        # Thikonov regularization, dinstinguish between 'X' and 'F' cases
        # as size of corresponding z is different
        # alpha > 0: Thikonov on the gradient of z
        if alpha > 0:
            if self.__id__ == 'X':
                self.lz_gpu = shock.laplace_stack_gpu(z_gpu, mode='same')

            elif self.__id__ == 'F':        
                self.lz_gpu = gputools.laplace_gpu(z_gpu, mode='same')

            obj += 0.5*alpha*(cua.dot(z_gpu, self.lz_gpu, dtype=np.float64))

        # beta  > 0: Thikonov on z
        if beta > 0:
            obj += 0.5*beta*(cua.dot(z_gpu, z_gpu,dtype=np.float64))
                
        return obj.get()
            
    def cnvinv_gradfun(self, z, sz, y_gpu, alpha=0., beta=0.):
        """
        Computes gradient used for 'lbfgsb' mode of deconv method.
        See deconv for details.
        """
        
        if z.__class__ == np.ndarray:
            z = np.array(np.reshape(z,sz)).astype(np.float32)
            z_gpu = cua.to_gpu(z)            

        grad_gpu =  self.cnvtp(self.res_gpu)
        
        # Thikonov regularization
        # alpha > 0: Thikonov on the gradient of z
        if alpha > 0:
            grad_gpu += alpha * self.lz_gpu

        # beta  > 0: Thikonov on z
        if beta > 0:
            grad_gpu += beta * z_gpu

        grad = -np.real(grad_gpu.get())
        grad = grad.flatten()
        return grad.astype(np.float64)

    def fft(self, y_gpu, batch_size):
        """
        Computes FFT with precomputed FFT plan
        """
        
        step_size = int(np.floor(float(2**15)/y_gpu.shape[1]))
        for i in range(0,batch_size,step_size):
            this_batch_size = min(step_size, batch_size-i)
            self.plan.execute(
                int(y_gpu.gpudata) +
                y_gpu.dtype.itemsize*i*y_gpu.shape[1]*y_gpu.shape[2],
                batch=this_batch_size)


if __name__ == '__main__':
    
    #CUDA_DEVICE = 2

    # Load other libraries 
    import pycuda.autoinit
    import pycuda.gpuarray as cua
    import pylab as pl
    import numpy as np
    import scipy 
    import time

    # Load own libraries
    import gputools
    import imagetools
    import olaGPU as ola
    
    x_rgb = np.array(pl.imread('lena.png')).astype(np.float32)
    x     = imagetools.rgb2gray(x_rgb)
    x_gpu = cua.to_gpu(x)

    sx      = x.shape
    csf     = (5,5)
    overlap = 0.5

    f  = pl.imread('f.png').astype(np.float32)
    f  = imagetools.rgb2gray(f)
    f  = scipy.misc.imresize(f,(17,17)).astype(np.float32)
    f /= f.sum()
    sf = f.shape
    fs = np.tile(f, (np.prod(csf), 1, 1))
    fs_gpu = cua.to_gpu(fs)

    print "-------------------"
    print "Create windows"
    start = time.clock()
    winaux = imagetools.win2winaux(sx, csf, overlap)

    # Use windows from matlab to analyse window artifacts
    #import scipy.io as scio
    #W = scio.loadmat('../olamat/test_olamat/W.mat')
    #W = W['ws'].flatten()
    #ws = np.zeros(winaux.ws_gpu.shape)
    #for i in np.arange(0,len(W)):
    #    ws[i] = W[i]
    #ws = ws.astype(np.float32)
    #winaux.ws_gpu = cua.to_gpu(ws)
    
    print "Time elapsed: %.4f" % (time.clock()-start)
    print "-------------------"
    print ""

    print "-------------------"
    print "Create X"
    start = time.clock()
    X = ola.OlaGPU(x_gpu, sf, mode='valid', winaux=winaux)
    print "Time elapsed: %.4f" % (time.clock()-start)
    print "-------------------"
    print ""

    print "-------------------"
    print "Compute X.cnv "
    start = time.clock()
    yX_gpu = X.cnv(fs_gpu)
    print "Time elapsed: %.4f" % (time.clock()-start)
    print "-------------------"
    print ""

    print "-------------------"
    print "Compute X.cnvtp "
    start = time.clock()
    xhat_gpu = X.cnvtp(yX_gpu)
    print "Time elapsed: %.4f" % (time.clock()-start)
    print "-------------------"
    print ""

    print "-------------------"
    print "Create F"
    start = time.clock()
    F = ola.OlaGPU(fs, sx, mode='valid', winaux=winaux)
    print "Time elapsed: %.4f" % (time.clock()-start)
    print "-------------------"
    print ""

    print "-------------------"
    print "Compute F.cnv "
    start = time.clock()
    yF_gpu = F.cnv(x_gpu)
    print "Time elapsed: %.4f" % (time.clock()-start)
    print "-------------------"
    print ""

    print "-------------------"
    print "Compute F.cnvtp "
    print "-------------------"
    start = time.clock()
    fhat_gpu = F.cnvtp(yF_gpu)
    print "Time elapsed: %.4f" % (time.clock()-start)
    print ""

    print "-------------------"
    print "Create invariant Fi"
    start = time.clock()
    Fi = cnv.CnvGPU(f, sx, mode='valid')
    print "Time elapsed: %.4f" % (time.clock()-start)
    print "-------------------"
    print ""

    print "-------------------"
    print "Compute Fi.cnv "
    start = time.clock()
    yi_gpu = Fi.cnv(x_gpu)
    print "Time elapsed: %.4f" % (time.clock()-start)
    print "-------------------"
    print ""

    print "-------------------"
    print "Copy to CPU "
    start = time.clock()
    yF = yF_gpu.get()
    yX = yX_gpu.get()
    yi = yi_gpu.get()
    print "Time elapsed: %.4f" % (time.clock()-start)
    print "-------------------"
    print ""

    diff = 1
    if diff:
        pl.figure(1)
        pl.imshow(yF-yi);
        pl.show()
        pl.title("Difference between olaGPU and cnvGPU")

    direct = 1
    if direct:
        print "-------------------"
        print "Direct deconvolution"
        start = time.clock()
        xhat_gpu = F.deconv(yF, mode = 'direct', alpha = 0.05,beta = 0.01)
        print "Time elapsed: %.4f" % (time.clock()-start)
        print "-------------------"
        print ""
     
        xhat = xhat_gpu.get()
        pl.figure(2)
        pl.imshow(xhat)
        pl.title("Result of direct deconvolution")
        pl.show()

    gdirect = 1
    if gdirect:
        print "-------------------"
        print "Gdirect deconvolution"
        start = time.clock()
        fhat_gpu = X.deconv(yX_gpu, mode = 'gdirect', alpha = 0.01,beta = 0.01)
        print "Time elapsed: %.4f" % (time.clock()-start)
        print "-------------------"
        print ""
     
        fhat = fhat_gpu.get()
        pl.figure(3)
        imagetools.cellplot(fhat, csf);
        pl.title("Result of gdirect deconvolution")
        pl.show()

    sparse = 1
    if sparse:
        print "-------------------"
        print "Sparse deconvolution"
        start = time.clock()
        xhat_gpu = F.deconv(yF_gpu, mode = 'sparse', alpha = 0.0001)
        print "Time elapsed: %.4f" % (time.clock()-start)
        print "-------------------"
        print ""
     
        xhat = xhat_gpu.get()
        pl.figure(4)
        pl.imshow(xhat)
        pl.title("Result of sparse deconvolution")
        pl.show()

    rgb = 0
    if rgb:
        y_rgb = np.zeros((yF_gpu.shape[0], yF_gpu.shape[1], 3))
        for i in range(3):
            y_rgb[...,i] = F.cnv(x_rgb[...,i])

        print "-------------------"
        print "Sparse deconvolution"
        start = time.clock()
        xhat_rgb = F.deconv_rgb(y_rgb, mode = 'sparse', alpha = 0.0001)
        print "Time elapsed: %.4f" % (time.clock()-start)
        print "-------------------"
        print ""
     
        pl.figure(3)
        pl.imshow(xhat_rgb)
        pl.title("Result of rgb deconvolution")
        pl.show()

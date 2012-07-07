#!/usr/bin/env python
# Needs to be fixed in chop_pad_GPU: if sx != csf * sw, artifacts arise
# Example: lena with csf = (4,4)
# in pad_cpu2gpu replace cua.zeros with cua.empty

#CUDA_DEVICE = 2
import pycuda.driver as cu
import pycuda.autoinit
import numpy as np
import pycuda.gpuarray as cua
import pycuda.driver as cu
import pycuda.elementwise as cuelement
from pycuda.compiler import compile
from pycuda.compiler import SourceModule
import pylab
import scipy.io
import scipy.misc
import scipy.ndimage.interpolation
import scipy.signal
import scipy.sparse
import time

import gputools
import imagetools
import optGPU

cubin = compile(open('gputools.cu').read(), keep=True)
edgetaper_code = open('edgetaper.cu').read()
kernel_code = open('kernel.cu').read()
resize_code = open('resize.cu').read()
projection_code = open('projection.cu').read()

def _generate_preproc(dtype, shape=None):
  
  if dtype == np.float32:
    preproc = '#define real float\n'
  elif dtype == np.float64:
    preproc = '#define real double\n'

  if shape != None:
    preproc += '#define ROWS %d\n' % shape[0]
    preproc += '#define COLS %d\n' % shape[1]

  return preproc

class BasisGpu:

  def __init__(self, lens_file=None, lens_psf_size=None, lens_grid_size=None):

    if lens_file:
      self.lens = True
      grid = scipy.misc.imread(lens_file, flatten=True)
      if np.max(grid) > 255:
        grid /= 2**(16-1)
      else:
        grid /= 255
      self.grid_gpu = cu.matrix_to_array(grid, 'C')
      self.lens_psf_size = lens_psf_size
      self.lens_grid_size = lens_grid_size
    else:
      self.lens = False

  def _psf2params(self):
    self.max_trans = np.floor(np.max(self.psf_size)/2.)-1    
    R = self.im_size-self.psf_size
    R = np.linalg.norm(R)
    dphi = 3*(np.sqrt(2.)/(2*np.pi*R) * 2*np.pi)
    self.max_phi = dphi * self.max_trans

    self.shape = np.int_(np.array([self.max_trans*2+1, self.max_trans*2+1,
                                  self.max_trans*2+1]))
    self.params = np.mgrid[-self.max_trans:self.max_trans+0.1,
                           -self.max_trans:self.max_trans+0.1,
                           -self.max_phi:self.max_phi+dphi/10.:dphi]
    self.params = self.params.T.reshape((np.prod(self.shape),3))

  def psf_in_cube(self, index):
    if np.array(index).size == 3:
     index = (index[2]*self.shape[0]*self.shape[1] +
              index[1]*self.shape[0] + index[0])
    psf = self.basis_host[index].toarray()
    psf = psf.reshape(
      (self.grid_size[0],self.grid_size[1],self.psf_size[0],self.psf_size[1]))
    psf = np.hstack(np.hstack(psf[:]))
    return psf

  def project(self, fs_gpu, lamda=1e2, eta=0., alpha = 2/3,
              sparsity=None, maxfun=20, mode=None, show=False):
    """
    Estimates weighting vector (w s.t. w>=0) by minimizing
    min |B*w - f|^2 + lamda * |w|^0.66 + eta * |grad(w)|^2
    """

    if fs_gpu.__class__ != np.ndarray:
      fs = fs_gpu.get()
    else:
      fs = fs_gpu

    # Naive projection
    wproj = self.basis_host.dot(fs.reshape((fs.size,1))).astype(np.float32)
    #wproj /= float(self._intern_shape[1])
    wproj = np.reshape(wproj, self.shape)
    
    if sparsity:
      wproj = imagetools.sparsify(wproj, sparsity)

    if mode == None:
      return cua.to_gpu(wproj)

    elif mode == 'HPQ_L1':
      if fs_gpu.__class__ == np.ndarray:
        self.data_gpu = cua.to_gpu(fs_gpu)
      else:
        self.data_gpu = fs_gpu

  def estimate_weight(self, w_gpu, u_gpu, beta):
    """
    Solves the following problem by projected Barzilai-Borwein
    """
    # Optimisation parameters
    options = optGPU.Solopt()
    options.maxiter = self.maxfun
    options.verbose = 0
    
    # Optimisation via PBB
    self.u_gpu = u_gpu
    self.beta  = beta
    
    pbb = optGPU.PBB(self, w_gpu, options)

    return pbb.result

  def compute_obj(self, w_gpu):

    self.dfs_gpu = 1. * (self.weight(w_gpu) - self.data_gpu)
    res =    0.5 * self.lamda * cua.dot(self.dfs_gpu, self.dfs_gpu) 
    reg = (  0.5 * self.beta  * cua.dot(w_gpu - self.u_gpu,
                                        w_gpu - self.u_gpu))

    if self.eta:
      reg += 0.5 * self.eta * cua.dot(w_gpu, laplace3d_gpu(w_gpu))

    return res + reg

  def compute_grad(self, w_gpu):

    self.dfs_gpu = 1. * (self.weight(w_gpu) - self.data_gpu)
    gw_gpu = (  self.lamda * self.project(self.dfs_gpu)
              + self.beta  * (w_gpu - self.u_gpu))

    if self.eta:
      gw_gpu += self.eta * laplace3d_gpu(w_gpu)
      
    return gw_gpu  

  def set_params(self, psf_size, grid_size, im_size, params=None):

    # generate grid
    psf_size = np.array(psf_size)
    grid_size = np.array(grid_size)
    im_size = np.array(im_size)
    self.psf_size = psf_size + (1 - np.mod(psf_size, 2))
    self.grid_size = grid_size
    self.im_size = im_size

    if params != None:
      self.params = params
      self.shape = (params.size / 3,)
    else:
      self._psf2params()
    
    if not self.lens:
      grid = np.zeros(self.psf_size, dtype=np.float32)
      grid[(self.psf_size[0]-1)/2, (self.psf_size[1]-1)/2] = 1.
      grid = np.tile(grid, self.grid_size)
      self.lens_psf_size = self.psf_size
      #lens_grid_size = (1,1)
      self.lens_grid_size = self.grid_size
      self.grid_gpu = cu.matrix_to_array(grid, 'C')

    params_count = np.uint32(self.params.size / 3)
    params_gpu = cu.matrix_to_array(self.params.astype(np.float32), 'C')

    #self.output_size = np.array(self.grid_size)*np.array(self.psf_size)
    output_size = np.array((np.prod(self.grid_size),
                            self.psf_size[0], self.psf_size[1]))

    preproc = '#define BLOCK_SIZE 0\n' #_generate_preproc(basis_gpu.dtype)
    mod = SourceModule(preproc + basis_code, keep=True)

    in_tex = mod.get_texref('in_tex')
    in_tex.set_array(self.grid_gpu)
    in_tex.set_filter_mode(cu.filter_mode.LINEAR)
    #in_tex.set_flags(cu.TRSF_NORMALIZED_COORDINATES)

    params_tex = mod.get_texref('params_tex')
    params_tex.set_array(params_gpu)
    offset = ((np.array(self.im_size) - np.array(grid.shape)) /
                   np.array(self.grid_size).astype(np.float32))
    offset = np.float32(offset)
    grid_scale = ((np.array(self.lens_grid_size) - 1) /
                       (np.array(self.grid_size) - 1).astype(np.float32))
    grid_scale = np.float32(grid_scale)

    block_size = (16,16,1)
    gpu_grid_size = (int(np.ceil(float(np.prod(output_size))/block_size[0])),
                     int(np.ceil(float(params_count)/block_size[1])))

    basis_gpu = cua.empty((int(params_count),
                           int(output_size[0]), int(output_size[1]),
                           int(output_size[2])), np.float32)
    #self.basis_host = cu.pagelocked_empty((int(params_count),
    #    int(output_size[0]), int(output_size[1]), int(output_size[2])),
    #    np.float32, mem_flags=cu.host_alloc_flags.DEVICEMAP)

    basis_fun_gpu = mod.get_function("basis")

    basis_fun_gpu(basis_gpu.gpudata,
                  np.uint32(np.prod(output_size)),
                  np.uint32(self.grid_size[1]),
                  np.uint32(self.psf_size[0]), np.uint32(self.psf_size[1]),
                  np.uint32(self.im_size[0]), np.uint32(self.im_size[1]),
                  offset[0], offset[1],
                  grid_scale[0], grid_scale[1],
                  np.uint32(self.lens_psf_size[0]),
                  np.uint32(self.lens_psf_size[1]),
                  params_count,
                  block=block_size, grid=gpu_grid_size)

    self.basis_host = basis_gpu.get()
    self._intern_shape = self.basis_host.shape
    self.basis_host = self.basis_host.reshape((self._intern_shape[0],
        self._intern_shape[1]*self._intern_shape[2]*self._intern_shape[3]))
    self.basis_host = scipy.sparse.csr_matrix(self.basis_host)

  def weight(self, weights):
    if weights.__class__ != np.ndarray:
      weights = weights.get()
    #result = self.basis_host.transpose().dot(
    #  scipy.sparse.csr_matrix(weights.reshape((weights.size,1))))
    result = self.basis_host.transpose().dot(
      weights.reshape((weights.size,1))).astype(np.float32)
    result = result.reshape(
      (self._intern_shape[1],self._intern_shape[2],self._intern_shape[3]))
    #result = gputools.normalize(result)
    #result = gputools.center(result)
    
    self.fs_gpu = cua.to_gpu(result)
    return self.fs_gpu
  
def bfilter_gpu(y_gpu, w=5, sigma_d=3, sigma_r=0.1):
  """Two dimensional bilateral filtering.

  This function implements 2-D bilateral filtering using the method
  outlined in:
      C. Tomasi and R. Manduchi. Bilateral Filtering for 
      Gray and Color Images. In Proceedings of the IEEE 
      International Conference on Computer Vision, 1998.
  This operation is in place.

  Args:
      y_gpu: Input image on GPU.
      w: Half-size of the filter.
      sigma_d: Spatial domain deviation.
      sigma_r: Intensity domain deviation.

  Returns:
      Nothing, but modifies y_gpu in place.    
  """

  shape = np.array(y_gpu.shape).astype(np.uint32)
  dtype = y_gpu.dtype
  block_size = (16,16,1)
  grid_size = (int(np.ceil(float(shape[1])/block_size[0])),
               int(np.ceil(float(shape[0])/block_size[1])))
  shared_size = int((2*w+block_size[0])*(2*w+block_size[1])*dtype.itemsize)

  preproc = _generate_preproc(dtype, shape)
  mod = SourceModule(preproc + kernel_code, keep=True)

  bfilter_gpu = mod.get_function("bfilter")

  bfilter_gpu(y_gpu.gpudata, np.int32(w), np.float32(sigma_d),
              np.float32(sigma_r),
              block=block_size, grid=grid_size, shared=shared_size)
  
  
def center(f_gpu):

  if f_gpu.__class__ == cua.GPUArray:
    f = f_gpu.get()
  else:
    f = f_gpu

  center = centerofmass(f)
  center -= (np.array(f[0].shape)-1)/2.

  if len(f.shape) == 3:
    for i in range(f.shape[0]):
      f[i] = scipy.ndimage.interpolation.shift(f[i], -np.round(center))
  
  if f_gpu.__class__ == cua.GPUArray:
    return cua.to_gpu(f)
  else:
    return f
  

def centerofmass(f, support=False):

  ftmp = f.copy()
  if support:
    ftmp[f>0] = 1

  ftmp = normalize(ftmp)

  if len(f.shape) == 3:
    cx = np.mean(np.sum((np.tensordot(np.arange(f.shape[1]),f,(0,1))),1))
    cy = np.mean(np.sum((np.tensordot(np.arange(f.shape[2]),f,(0,2))),1))
  return np.array([cx,cy])


def chop_mod_pad_GPU(x, ws_gpu, csf, sw, nhop, sz=None, offset=(0,0), dtype='real'):
    
    sx  = x.shape

    if sz == None:
        sz = 2**np.ceil(np.log2(sw))

    block_size = (32,32,1)   
    grid_size = (int(np.ceil(np.float32(np.prod(csf)*sz[0])/block_size[0])),
                 int(np.ceil(np.float32(sz[1])/block_size[1])))

    sxp = np.array(nhop*csf+sw-nhop)
    sxp = tuple((int(sxp[0]),int(sxp[1])))

    x_gpu = pad_cpu2gpu(x, sxp, dtype='real')
    
    if dtype == 'real':
        mod = cu.module_from_buffer(cubin)
        chop_mod_pad_Kernel = mod.get_function("chop_mod_pad_Kernel")
        xs_gpu = cua.empty(tuple((int(np.prod(csf)), int(sz[0]),int(sz[1]))),
                           np.float32)
    elif dtype == 'complex':
        mod = cu.module_from_buffer(cubin)
        chop_mod_pad_Kernel = mod.get_function("chop_mod_pad_ComplexKernel")        
        xs_gpu = cua.empty(tuple((int(np.prod(csf)), int(sz[0]),int(sz[1]))),
                           np.complex64)

    sz = xs_gpu.shape        
    chop_mod_pad_Kernel(xs_gpu.gpudata, np.int32(sz[1]),
                               np.int32(sz[2]), np.int32(sz[0]),
                               x_gpu.gpudata,
                               np.int32(sxp[0]), np.int32(sxp[1]),
                               ws_gpu.gpudata,
                               np.int32(sw[0]), np.int32(sw[1]),
                               np.int32(offset[0]), np.int32(offset[1]),
                               np.int32(csf[0]), np.int32(csf[1]),
                               np.int32(nhop[0]), np.int32(nhop[1]),
                               block=block_size, grid=grid_size)

    return xs_gpu
  

def chop_pad_GPU(x, csf, sw, nhop, sz=None, offset=(0,0), dtype='real'):
    
    sx  = x.shape

    if sz == None:
        sz = 2**np.ceil(np.log2(sw))

    block_size = (32,32,1)   
    grid_size = (int(np.ceil(np.float32(np.prod(csf)*sz[0])/block_size[0])),
                 int(np.ceil(np.float32(sz[1])/block_size[1])))

    sxp = np.array(nhop*csf+sw-nhop)
    sxp = tuple((int(sxp[0]),int(sxp[1])))

    x_gpu = pad_cpu2gpu(x, sxp, dtype='real')
    
    if dtype == 'real':
        mod = cu.module_from_buffer(cubin)
        chop_pad_Kernel = mod.get_function("chop_pad_Kernel")
        xs_gpu = cua.empty(tuple((int(np.prod(csf)), int(sz[0]),int(sz[1]))),
                           np.float32)
    elif dtype == 'complex':
        mod = cu.module_from_buffer(cubin)
        chop_pad_Kernel = mod.get_function("chop_pad_ComplexKernel")        
        xs_gpu = cua.empty(tuple((int(np.prod(csf)), int(sz[0]),int(sz[1]))),
                           np.complex64)

    sz = xs_gpu.shape        
    chop_pad_Kernel(xs_gpu.gpudata, np.int32(sz[1]),
                               np.int32(sz[2]), np.int32(sz[0]), x_gpu.gpudata,
                               np.int32(sxp[0]), np.int32(sxp[1]),
                               np.int32(sw[0]), np.int32(sw[1]),
                               np.int32(offset[0]), np.int32(offset[1]),
                               np.int32(csf[0]), np.int32(csf[1]),
                               np.int32(nhop[0]), np.int32(nhop[1]),
                               block=block_size, grid=grid_size)

    return xs_gpu


def chop_pad_GPU_test(x, csf, sw, nhop, sz=None, offset=(0,0), dtype='real'):
    
    sx = x.shape

    if sz == None:
        sz = 2**np.ceil(np.log2(sw))

    block_size = (32,32,1)
    grid_size = (int(np.ceil(np.float32(sz[0]*sz[1])/block_size[0])),
                 int(np.ceil(np.float32(np.prod(csf))/block_size[1])))

    #print block_size
    #print grid_size
    #print csf

    sxp = np.array(nhop*csf+sw-nhop)
    sxp = ((int(sxp[0]), int(sxp[1])))

    x_gpu = pad_cpu2gpu(x, sxp, dtype='real')
    
    if dtype == 'real':
        mod = cu.module_from_buffer(cubin)
        chop_pad_Kernel = mod.get_function("chop_pad_Kernel_test")
        xs_gpu = cua.empty(((int(np.prod(csf)), int(sz[0]),int(sz[1]))),
                           np.float32)
    elif dtype == 'complex':
        mod = cu.module_from_buffer(cubin)
        chop_pad_Kernel = mod.get_function("chop_pad_ComplexKernel_test")     
        xs_gpu = cua.empty(((int(np.prod(csf)), int(sz[0]),int(sz[1]))),
                           np.complex64)
        
    sz = xs_gpu.shape        
    chop_pad_Kernel(xs_gpu.gpudata, np.int32(sz[1]),
                               np.int32(sz[2]), np.int32(sz[0]), x_gpu.gpudata,
                               np.int32(sxp[0]), np.int32(sxp[1]),
                               np.int32(sw[0]), np.int32(sw[1]),
                               np.int32(offset[0]), np.int32(offset[1]),
                               np.int32(csf[0]), np.int32(csf[1]),
                               np.int32(nhop[0]), np.int32(nhop[1]),
                               block=block_size, grid=grid_size)

    return xs_gpu

def isgreater_gpu(x_gpu, y_gpu):
  """
  Computes if x_gpu > y_gpu and gives back a mask with 0s and 1s.
  Note, that y_gpu can be a scalar value as well.
  """

  if ((y_gpu.__class__ == np.float) or 
      (y_gpu.__class__ == np.float32) or 
      (y_gpu.__class__ == np.float64)):

    val   = np.float32(y_gpu)
    y_gpu = cua.empty_like(x_gpu)   # gpu array containing threshold
    y_gpu.fill(val)
 
  zeros_gpu = cua.zeros_like(x_gpu)   # gpu array filled with zeros
  ones_gpu  = cua.empty_like(x_gpu)   # gpu array containing threshold
  ones_gpu.fill(np.float32(1.))        

  mask_gpu  = cua.if_positive(x_gpu > y_gpu, ones_gpu, zeros_gpu)

  del zeros_gpu
  del ones_gpu

  return mask_gpu

def clip_GPU(x_gpu, lb, ub):    

    cliplower_GPU(x_gpu, lb)
    clipupper_GPU(x_gpu, ub)


def cliplower_GPU(x_gpu, lb):    

    cliplower = cuelement.ElementwiseKernel(
        "float *x, float lb",
        "x[i] = x[i] < lb ? lb : x[i]",
        "cliplower")

    cliplower(x_gpu, lb)
    

def clipupper_GPU(x_gpu, ub):    

    clipupper = cuelement.ElementwiseKernel(
        "float *x, float ub",
        "x[i] = x[i] > ub ? ub : x[i]",
        "clipupper")

    clipupper(x_gpu, ub)
    

def comp_ola_deconv(fs_gpu, ys_gpu, L_gpu, alpha, beta):
    """
    Computes the division in Fourier space needed for direct deconvolution
    """
    
    sfft = fs_gpu.shape
    block_size = (16,16,1)   
    grid_size = (int(np.ceil(np.float32(sfft[0]*sfft[1])/block_size[0])),
                 int(np.ceil(np.float32(sfft[2])/block_size[1])))

    mod = cu.module_from_buffer(cubin)
    comp_ola_deconv_Kernel = mod.get_function("comp_ola_deconv_Kernel")

    z_gpu = cua.zeros(sfft, np.complex64)

    comp_ola_deconv_Kernel(z_gpu.gpudata,
                           np.int32(sfft[0]), np.int32(sfft[1]), np.int32(sfft[2]),
                           fs_gpu.gpudata, ys_gpu.gpudata, L_gpu.gpudata,
                           np.float32(alpha), np.float32(beta),
                           block=block_size, grid=grid_size)

    return z_gpu
  

def comp_ola_gdeconv(xx_gpu, xy_gpu, yx_gpu, yy_gpu, L_gpu, alpha, beta):
    """
    Computes the division in Fourier space needed for gdirect deconvolution
    """
    
    sfft = xx_gpu.shape
    block_size = (16,16,1)   
    grid_size = (int(np.ceil(np.float32(sfft[0]*sfft[1])/block_size[0])),
                 int(np.ceil(np.float32(sfft[2])/block_size[1])))

    mod = cu.module_from_buffer(cubin)
    comp_ola_gdeconv_Kernel = mod.get_function("comp_ola_gdeconv_Kernel")

    z_gpu = cua.zeros(sfft, np.complex64)

    comp_ola_gdeconv_Kernel(z_gpu.gpudata,
                            np.int32(sfft[0]), np.int32(sfft[1]), np.int32(sfft[2]),
                            xx_gpu, xy_gpu, yx_gpu, yy_gpu, L_gpu.gpudata,
                            np.float32(alpha), np.float32(beta),
                            block=block_size, grid=grid_size)

    return z_gpu
  

def crop_gpu2cpu(x_gpu, sz, offset=(0,0)):

    sfft = x_gpu.shape

    block_size = (16, 16 ,1)
    grid_size = (int(np.ceil(np.float32(sfft[1])/block_size[1])),
                 int(np.ceil(np.float32(sfft[0])/block_size[0])))

    if x_gpu.dtype == np.float32:
        mod = cu.module_from_buffer(cubin)
        cropKernel = mod.get_function("crop_Kernel")

    elif x_gpu.dtype == np.complex64:
        mod = cu.module_from_buffer(cubin)
        cropKernel = mod.get_function("crop_ComplexKernel")

    x_cropped_gpu = cua.empty(tuple((int(sz[0]),int(sz[1]))), np.float32)
        
    cropKernel(x_cropped_gpu.gpudata,   np.int32(sz[0]),       np.int32(sz[1]),
                       x_gpu.gpudata, np.int32(sfft[0]),     np.int32(sfft[1]),
                                    np.int32(offset[0]), np.int32(offset[1]),
                                    block=block_size   , grid=grid_size)

    return x_cropped_gpu
  

def comp_ola_sdeconv(gx_gpu, gy_gpu, xx_gpu, xy_gpu, Ftpy_gpu, f_gpu, L_gpu, alpha, beta, gamma=0):
    """
    Computes the division in Fourier space needed for sparse deconvolution
    """
    
    sfft = xx_gpu.shape
    block_size = (16,16,1)   
    grid_size = (int(np.ceil(np.float32(sfft[0]*sfft[1])/block_size[0])),
                 int(np.ceil(np.float32(sfft[2])/block_size[1])))

    mod = cu.module_from_buffer(cubin)
    comp_ola_sdeconv_Kernel = mod.get_function("comp_ola_sdeconv_Kernel")

    z_gpu = cua.zeros(sfft, np.complex64)

    comp_ola_sdeconv_Kernel(z_gpu.gpudata,
                            np.int32(sfft[0]), np.int32(sfft[1]), np.int32(sfft[2]),
                            gx_gpu.gpudata, gy_gpu.gpudata,
                            xx_gpu.gpudata, xy_gpu.gpudata, 
                            Ftpy_gpu.gpudata, f_gpu.gpudata, L_gpu.gpudata,
                            np.float32(alpha), np.float32(beta),
                            np.float32(gamma),
                            block=block_size, grid=grid_size)

    return z_gpu
    
    
def crop_stack_GPU(x, sz, offset=(0,0), dtype='real'):
    
    if x.__class__ == np.ndarray:
        x = np.array(x).astype(np.float32)
        x_gpu = cua.to_gpu(x)
    elif x.__class__ == cua.GPUArray:
        x_gpu = x

    sx = x_gpu.shape
    block_size = (16,16,1)   
    grid_size = (int(np.ceil(np.float32(sx[0]*sz[0])/block_size[0])),
                 int(np.ceil(np.float32(sz[1])/block_size[1])))

    sx_before = np.array([sx[1],sx[2]])
    sx_after  = np.array(sz)
    if any(np.array([sx[1],sx[2]])-(np.array(sz))<offset):
        raise IOError('Size missmatch: Size after - size before < offset')
    

    if dtype == 'real':

        if x_gpu.dtype != np.float32:
            x_gpu = x_gpu.real

        mod = cu.module_from_buffer(cubin)
        crop_stack_Kernel = mod.get_function("crop_stack_Kernel")

        xc_gpu = cua.zeros(tuple((int(sx[0]), int(sz[0]), int(sz[1]))), np.float32)

    if dtype == 'complex':
     
        mod = cu.module_from_buffer(cubin)
        crop_stack_Kernel = mod.get_function("crop_stack_ComplexKernel")
        xc_gpu = cua.empty(tuple((int(sx[0]), int(sz[0]), int(sz[1]))), np.complex64)
        
    crop_stack_Kernel(xc_gpu.gpudata, np.int32(sx[0]),
                                      np.int32(sz[0]),     np.int32(sz[1]),
                       x_gpu.gpudata, np.int32(sx[1]),     np.int32(sx[2]),
                                      np.int32(offset[0]), np.int32(offset[1]),
                                      block=block_size, grid=grid_size)
        
    return xc_gpu
  

def edgetaper_gpu(y_gpu, sf, win='barthann'):

  shape = np.array(y_gpu.shape).astype(np.uint32)
  dtype = y_gpu.dtype
  block_size = (16,16,1)
  grid_size = (int(np.ceil(float(shape[1])/block_size[0])),
               int(np.ceil(float(shape[0])/block_size[1])))

  # Ensure that sf is odd
  sf = sf+(1-np.mod(sf,2))
  wx = scipy.signal.get_window(win, sf[1])
  wy = scipy.signal.get_window(win, sf[0])
  maxw = wx.max() * wy.max()
  
  hsf = np.floor(sf/2)
  wx = (wx[0:hsf[1]] / maxw).astype(dtype)
  wy = (wy[0:hsf[0]] / maxw).astype(dtype)

  preproc = _generate_preproc(dtype, shape)
  preproc += '#define wx_size %d\n' % wx.size
  preproc += '#define wy_size %d\n' % wy.size
  mod = SourceModule(preproc + edgetaper_code, keep=True)
  edgetaper_gpu = mod.get_function("edgetaper")
  wx_gpu, wx_size = mod.get_global('wx')
  wy_gpu, wy_size = mod.get_global('wy')

  cu.memcpy_htod(wx_gpu, wx)
  cu.memcpy_htod(wy_gpu, wy)

  edgetaper_gpu(y_gpu, np.int32(hsf[1]), np.int32(hsf[0]),
                block=block_size, grid=grid_size)
  
  
def gamma_compress(image):
  """Do a gamma compression on the image.

  This function does a gamma compression on the input image.

  Args:
      image: Input image.

  Returns:
      Compressed image
  """
  
  result = image.copy()
  threshold = 0.0031308
  result[image<=threshold] = 12.92 * image[image<=threshold]
  result[image>threshold] = 1.055 * image[image>threshold] ** (1./2.4) - 0.055
  result[result<0] = 0
  result[result>1] = 1
  return result


def gamma_compress_gpu(y_gpu):
  """Do a gamma compression on the image.

  This function does an in-place gamma compression on the input image.

  Args:
      y_gpu: Input image on GPU.

  Returns:
      Nothing, but modifies y_gpu in place.
  """
  
  shape = np.array(y_gpu.shape).astype(np.uint32)
  dtype = y_gpu.dtype
  if len(shape) == 3:
    dim = int(shape[2])
  else:
    dim = 1
  block_size = (16,16,dim)
  grid_size = (int(np.ceil(float(shape[1])/block_size[0])),
               int(np.ceil(float(shape[0])/block_size[1])))

  preproc = _generate_preproc(dtype, shape)
  mod = SourceModule(preproc + kernel_code, keep=True)

  gamma_compress_fun = mod.get_function("gamma_compress")
  gamma_compress_fun(y_gpu.gpudata, block=block_size, grid=grid_size)
  

def gamma_uncompress(image):
  """Do a gamma decompression on the image.

  This function does a gamma decompression on the input image.

  Args:
      image: Input image.

  Returns:
      Decompressed image
  """
  
  result = image.copy()
  threshold = 0.04045
  result[image<=threshold] = image[image<=threshold] / 12.92
  result[image>threshold] = ((image[image>threshold] + 0.055) / 1.055) ** 2.4
  result[result<0] = 0
  result[result>1] = 1
  return result


def gamma_uncompress_gpu(image):
  """Do a gamma decompression on the image.

  This function does an in-place gamma decompression on the input image.

  Args:
      y_gpu: Input image on GPU.

  Returns:
      Nothing, but modifies y_gpu in place.
  """
  
  shape = np.array(y_gpu.shape).astype(np.uint32)
  dtype = y_gpu.dtype
  if len(shape) == 3:
    dim = int(shape[2])
  else:
    dim = 1
  block_size = (16,16,dim)
  grid_size = (int(np.ceil(float(shape[1])/block_size[0])),
               int(np.ceil(float(shape[0])/block_size[1])))
  
  preproc = _generate_preproc(dtype, shape)
  mod = SourceModule(preproc + kernel_code, keep=True)
  
  gamma_uncompress_fun = mod.get_function("gamma_uncompress")
  gamma_uncompress_fun(y_gpu.gpudata, block=block_size, grid=grid_size) 


def generate_basis_gpu(psf_size, grid_size, im_size, params,
                       lens_file=None, lens_psf_size=None, lens_grid_size=None):

  # generate grid
  psf_size = psf_size + (1 - np.mod(psf_size, 2))
  
  if lens_file:
    grid = scipy.misc.imread(lens_file, flatten=True)
    if np.max(grid) > 255:
      grid /= 2**(16-1)
    else:
      grid /= 255
  else:
    grid = np.zeros(psf_size, dtype=np.float32)
    grid[(psf_size[0]-1)/2, (psf_size[1]-1)/2] = 1.
    grid = np.tile(grid, grid_size)
    lens_psf_size = psf_size
    #lens_grid_size = (1,1)
    lens_grid_size = grid_size

  grid_gpu = cu.matrix_to_array(grid, 'C')

  # generate parameters of basis functions
  #p = max(1, np.floor(psf_size[0] / 2))
  #p = min(8, p)
  #dp = min(45. / np.floor(psf_size * grid_size / 2))
  #dp = min(0.8, dp)
  #dp = np.radians(dp)
  #p = np.radians(p)
  #l = max(1, np.floor(psf_size[0] / 2))
  #l = np.ceil(l / 2)
  #params = np.mgrid[-l:l+1, -l:l+1, -p:p+dp/10:dp].astype(np.float32).T
  #params = params.reshape(params.size / 3, 3)
  params_gpu = cu.matrix_to_array(params.astype(np.float32), 'C')

  block_size = (16,16,1)
  output_size = np.array((np.prod(np.array(grid_size)),psf_size[0],psf_size[1]))
  gpu_grid_size = (int(np.ceil(float(np.prod(output_size))/block_size[0])),
                   int(np.ceil(float(params.size/3)/block_size[1])))

  basis_gpu = cua.empty((params.size/3,
                         int(output_size[0]), int(output_size[1]),
                         int(output_size[2])), np.float32)

  preproc = '#define BLOCK_SIZE 0\n' #_generate_preproc(basis_gpu.dtype)
  mod = SourceModule(preproc + basis_code, keep=True)
  basis_fun_gpu = mod.get_function("basis")

  in_tex = mod.get_texref('in_tex')
  in_tex.set_array(grid_gpu)
  in_tex.set_filter_mode(cu.filter_mode.LINEAR)
  #in_tex.set_flags(cu.TRSF_NORMALIZED_COORDINATES)

  params_tex = mod.get_texref('params_tex')
  params_tex.set_array(params_gpu)
  offset = ((np.array(im_size) - np.array(grid.shape)) /
            np.array(grid_size).astype(np.float32))
  offset = np.float32(offset)
  grid_scale = ((np.array(lens_grid_size) - 1) /
                (np.array(grid_size) - 1).astype(np.float32))
  grid_scale = np.float32(grid_scale)
  #psf_scale = ((np.array(lens_psf_size) - 1) /
  #             (np.array(psf_size) - 1).astype(np.float32))
  #psf_scale = np.float32(psf_scale)

  basis_fun_gpu(basis_gpu.gpudata,
                np.uint32(np.prod(output_size)), np.uint32(grid_size[1]),
                np.uint32(psf_size[0]), np.uint32(psf_size[1]),
                np.uint32(im_size[0]), np.uint32(im_size[1]),
                offset[0], offset[1],
                grid_scale[0], grid_scale[1],
                np.uint32(lens_psf_size[0]), np.uint32(lens_psf_size[1]),
                np.uint32(params.size/3),
                block=block_size, grid=gpu_grid_size)

  return basis_gpu


def gradient_gpu(y_gpu, mode='valid'):

  shape = np.array(y_gpu.shape).astype(np.uint32)
  dtype = y_gpu.dtype
  block_size = (16,16,1)
  grid_size = (int(np.ceil(float(shape[1])/block_size[0])),
               int(np.ceil(float(shape[0])/block_size[1])))
  shared_size = int((1+block_size[0])*(1+block_size[1])*dtype.itemsize)

  preproc = _generate_preproc(dtype, shape)
  mod = SourceModule(preproc + kernel_code, keep=True)

  if mode == 'valid':
    gradient_gpu = mod.get_function("gradient_valid")

    gradx_gpu = cua.empty((y_gpu.shape[0], y_gpu.shape[1]-1), y_gpu.dtype)
    grady_gpu = cua.empty((y_gpu.shape[0]-1, y_gpu.shape[1]), y_gpu.dtype)

  if mode == 'same':
    gradient_gpu = mod.get_function("gradient_same")

    gradx_gpu = cua.empty((y_gpu.shape[0], y_gpu.shape[1]), y_gpu.dtype)
    grady_gpu = cua.empty((y_gpu.shape[0], y_gpu.shape[1]), y_gpu.dtype)
    
  gradient_gpu(gradx_gpu.gpudata, grady_gpu.gpudata, y_gpu.gpudata,
               block=block_size, grid=grid_size, shared=shared_size)

  return (gradx_gpu, grady_gpu)


def impad_gpu(y_gpu, sf):

  sf = np.array(sf)
  shape = (np.array(y_gpu.shape) + sf).astype(np.uint32)
  dtype = y_gpu.dtype
  block_size = (16,16,1)
  grid_size = (int(np.ceil(float(shape[1])/block_size[0])),
               int(np.ceil(float(shape[0])/block_size[1])))

  preproc = _generate_preproc(dtype, shape)
  mod = SourceModule(preproc + kernel_code, keep=True)

  padded_gpu = cua.empty((int(shape[0]), int(shape[1])), dtype)
  impad_fun = mod.get_function("impad")

  upper_left = np.uint32(np.floor(sf / 2.))
  original_size = np.uint32(np.array(y_gpu.shape))

  impad_fun(padded_gpu.gpudata, y_gpu.gpudata,
            upper_left[1], upper_left[0],
            original_size[0], original_size[1],
            block=block_size, grid=grid_size)

  return padded_gpu


def laplace_gpu(y_gpu, mode='valid'):

  shape = np.array(y_gpu.shape).astype(np.uint32)
  dtype = y_gpu.dtype
  block_size = (16,16,1)
  grid_size = (int(np.ceil(float(shape[1])/block_size[0])),
               int(np.ceil(float(shape[0])/block_size[1])))
  shared_size = int((2+block_size[0])*(2+block_size[1])*dtype.itemsize)

  preproc = _generate_preproc(dtype, shape)
  mod = SourceModule(preproc + kernel_code, keep=True)

  if mode == 'valid':
    laplace_fun_gpu = mod.get_function("laplace_valid")
    laplace_gpu = cua.empty((y_gpu.shape[0]-2, y_gpu.shape[1]-2), y_gpu.dtype)

  if mode == 'same':
    laplace_fun_gpu = mod.get_function("laplace_same")
    laplace_gpu = cua.empty((y_gpu.shape[0], y_gpu.shape[1]), y_gpu.dtype)
    
  laplace_fun_gpu(laplace_gpu.gpudata, y_gpu.gpudata,
                  block=block_size, grid=grid_size, shared=shared_size)

  return laplace_gpu


def laplace_stack_gpu(y_gpu, mode='valid'):
  """
  This funtion computes the Laplacian of each slice of a stack of images
  """
  shape = np.array(y_gpu.shape).astype(np.uint32)
  dtype = y_gpu.dtype
  block_size = (6,int(np.floor(512./6./float(shape[0]))),int(shape[0]))
  grid_size = (int(np.ceil(float(shape[1])/block_size[0])),
               int(np.ceil(float(shape[0])/block_size[1])))
  shared_size = int((2+block_size[0])*(2+block_size[1])*(2+block_size[2])
                    *dtype.itemsize)

  preproc = _generate_preproc(dtype, (shape[1],shape[2]))
  mod = SourceModule(preproc + kernel_code, keep=True)

  laplace_fun_gpu = mod.get_function("laplace_stack_same")
  laplace_gpu = cua.empty((y_gpu.shape[0], y_gpu.shape[1], y_gpu.shape[2]),
                          y_gpu.dtype)
    
  laplace_fun_gpu(laplace_gpu.gpudata, y_gpu.gpudata,
                  block=block_size, grid=grid_size, shared=shared_size)
  
  return laplace_gpu


def laplace3d_gpu(y_gpu):

  shape = np.array(y_gpu.shape).astype(np.uint32)
  dtype = y_gpu.dtype
  block_size = (6,int(np.floor(512./6./float(shape[0]))),int(shape[0]))
  grid_size = (int(np.ceil(float(shape[1])/block_size[0])),
               int(np.ceil(float(shape[0])/block_size[1])))
  shared_size = int((2+block_size[0])*(2+block_size[1])*(2+block_size[2])
                    *dtype.itemsize)

  preproc = _generate_preproc(dtype, (shape[1],shape[2]))
  mod = SourceModule(preproc + kernel_code, keep=True)

  laplace_fun_gpu = mod.get_function("laplace3d_same")
  laplace_gpu = cua.empty((y_gpu.shape[0], y_gpu.shape[1], y_gpu.shape[2]),
                          y_gpu.dtype)
    
  laplace_fun_gpu(laplace_gpu.gpudata, y_gpu.gpudata,
                  block=block_size, grid=grid_size, shared=shared_size)

  return laplace_gpu


def modify_sparse23_gpu(y_gpu, beta):

  shape = np.array(y_gpu.shape).astype(np.uint32)
  gpu_shape = np.array([np.prod(shape),np.prod(shape)])
  gpu_shape = np.uint32(np.ceil(np.sqrt(gpu_shape)))
  dtype = y_gpu.dtype
  block_size = (16,16,1)
  grid_size = (int(np.ceil(float(gpu_shape[1])/block_size[0])),
               int(np.ceil(float(gpu_shape[0])/block_size[1])))

  preproc = _generate_preproc(dtype, np.array(grid_size)
                              * np.array(block_size)[0:1])
  mod = SourceModule(preproc + kernel_code, keep=True)

  modify_alpha23_fun = mod.get_function("modify_alpha23")

  modify_alpha23_fun(y_gpu.gpudata, np.float32(beta), np.uint32(np.prod(shape)),
                     block=block_size, grid=grid_size)
  

def modify_sparse_gpu(y_gpu, beta, alpha=2/3):

  shape = np.array(y_gpu.shape).astype(np.uint32)
  gpu_shape = np.array([np.prod(shape),np.prod(shape)])
  gpu_shape = np.uint32(np.ceil(np.sqrt(gpu_shape)))
  dtype = y_gpu.dtype
  block_size = (16,16,1)
  grid_size = (int(np.ceil(float(gpu_shape[1])/block_size[0])),
               int(np.ceil(float(gpu_shape[0])/block_size[1])))

  preproc = _generate_preproc(dtype, np.array(grid_size)
                              * np.array(block_size)[0:1])
  mod = SourceModule(preproc + kernel_code, keep=True)

  modify_alpha_fun = mod.get_function("modify_alpha")

  modify_alpha_fun(y_gpu.gpudata, np.float32(beta),
                   np.float32(alpha), np.uint32(np.prod(shape)),
                     block=block_size, grid=grid_size)


def normalize(f_gpu):

  if f_gpu.__class__ == cua.GPUArray:
    f = f_gpu.get()
  else:
    f = f_gpu

  if len(f.shape) == 3:
    for i in range(f.shape[0]):
      normval = f[i].sum()
      if normval > 1e-6:
        f[i] /= normval

  else:
    f /= f.sum()

  if f_gpu.__class__ == cua.GPUArray:
    return cua.to_gpu(f)
  else:
    return f
  

def ola_GPU(xs_gpu, sy, csf, hop):

    y_gpu = cua.empty(sy, np.float32)

    block_size = (16,16,1)   
    grid_size = (int(np.ceil(np.float32(sx[0]*sz[0])/block_size[1])),
                 int(np.ceil(np.float32(sz[1])/block_size[0])))

    mod = cu.module_from_buffer(cubin)
    copy_Kernel = mod.get_function("copy_Kernel")

    for i in range(csf[0]):
        for j in range(csf[1]):
            copy_Kernel(y_gpu,  np.uint32(sy[0]), np.uint32(sy[0]),
                        xs_gpu, np.uint32(sx[0]), np.uint32(sx[1]), np.uint32(sx[2]),
                        np.uint32(offset[0]), np.uint32(offset[1]), np.uint32(startrow), 
                        block=block_size, grid=grid_size)

    return np.real(y_gpu.get())
  

def ola_GPU_test(xs_gpu, csf, sw, nhop, offset=(0,0)):
    
    sxs = xs_gpu.shape

    sx = np.array(nhop*csf+sw-nhop)
    sx = ((int(sx[0]), int(sx[1])))

    block_size = (16,16,1)
    grid_size = (int(np.ceil(float(sx[1])/block_size[1])),
                 int(np.ceil(float(sx[0])/block_size[0])))

    if xs_gpu.dtype == np.float32:
        mod = cu.module_from_buffer(cubin)
        ola_Kernel = mod.get_function("ola_Kernel_test")
    elif xs_gpu.dtype == np.complex64:
        mod = cu.module_from_buffer(cubin)
        ola_Kernel = mod.get_function("ola_ComplexKernel_test")

    x_gpu = cua.zeros(sx, np.float32)
    ola_Kernel(x_gpu.gpudata, xs_gpu.gpudata,
               np.uint32(sx[0]), np.uint32(sx[1]),
               np.uint32(sxs[1]), np.uint32(sxs[2]),
               np.uint32(sw[0]), np.uint32(sw[1]),
               np.uint32(offset[0]), np.uint32(offset[1]),
               np.uint32(csf[0]), np.uint32(csf[1]),
               np.uint32(nhop[0]), np.uint32(nhop[1]),
               block=block_size, grid=grid_size)

    return x_gpu
  

def pad_cpu2gpu(x, sz, offset=(0,0), dtype='real'):

    block_size = (16, 16 ,1)
    grid_size = (int(np.ceil(np.float32(sz[1])/block_size[1])),
                 int(np.ceil(np.float32(sz[0])/block_size[0])))

    sx = x.shape

    if x.__class__ == np.ndarray:
        x  = np.array(x).astype(np.float32)        
        x_gpu = cua.to_gpu(x)        
    elif x.__class__ == cua.GPUArray:       
        x_gpu = x

    if dtype == 'real':

        mod = cu.module_from_buffer(cubin)
        zeroPadKernel = mod.get_function("zeroPadKernel")

        x_padded_gpu = cua.zeros(tuple((int(sz[0]),int(sz[1]))), np.float32)
        
        zeroPadKernel(x_padded_gpu.gpudata, np.int32(sz[0]),     np.int32(sz[1]),
                             x_gpu.gpudata, np.int32(sx[0]),     np.int32(sx[1]),
                                            np.int32(offset[0]), np.int32(offset[1]),
                                            block=block_size, grid=grid_size)
    elif dtype == 'complex':

        mod = cu.module_from_buffer(cubin)
        #mod = SourceModule(open('gputools.cu').read(), keep=True)
        zeroPadComplexKernel = mod.get_function("zeroPadComplexKernel")

        x_padded_gpu = cua.zeros(tuple((int(sz[0]),int(sz[1]))), np.complex64)
        
        zeroPadComplexKernel(x_padded_gpu.gpudata, np.int32(sz[0]),     np.int32(sz[1]),
                                    x_gpu.gpudata, np.int32(sx[0]),     np.int32(sx[1]),
                                                   np.int32(offset[0]), np.int32(offset[1]),
                                                   block=block_size, grid=grid_size)

    return x_padded_gpu
  

def pad_stack_GPU(x, sz, offset=(0,0), dtype='real'):

    if x.__class__ == np.ndarray:
        x = np.array(x).astype(np.float32)
        x_gpu = cua.to_gpu(x)
    elif x.__class__ == cua.GPUArray:
        x_gpu = x

    sx = x_gpu.shape

    block_size = (16,16,1)   
    grid_size = (int(np.ceil(np.float32(sx[0]*sz[0])/block_size[0])),
                 int(np.ceil(np.float32(sz[1])/block_size[1])))

    if dtype == 'real':

        if x_gpu.dtype != np.float32:
            x_gpu = x_gpu.real

        mod = cu.module_from_buffer(cubin)
        pad_stack_Kernel = mod.get_function("pad_stack_Kernel")

        xp_gpu = cua.empty(tuple((int(sx[0]), int(sz[0]), int(sz[1]))), np.float32)
        pad_stack_Kernel(x_gpu.gpudata, np.int32(sx[0]),
                                        np.int32(sx[1]),     np.int32(sx[2]),
                        xp_gpu.gpudata, np.int32(sz[0]),     np.int32(sz[1]),
                                        np.int32(offset[0]), np.int32(offset[1]),
                                        block=block_size, grid=grid_size)

    if dtype == 'complex':

        mod = cu.module_from_buffer(cubin)
        pad_stack_Kernel = mod.get_function("pad_stack_ComplexKernel")

        xp_gpu = cua.empty(tuple((int(sx[0]), int(sz[0]), int(sz[1]))), np.complex64)    
        pad_stack_Kernel(x_gpu.gpudata, np.int32(sx[0]),
                                        np.int32(sx[1]),     np.int32(sx[2]),
                        xp_gpu.gpudata, np.int32(sz[0]),     np.int32(sz[1]),
                                        np.int32(offset[0]), np.int32(offset[1]),
                                        block=block_size, grid=grid_size)
        
    return xp_gpu
  
  
def project_on_basis_gpu(fs_gpu, basis_gpu):

  basis_length = basis_gpu.shape[0]
  shape = np.array(fs_gpu.shape).astype(np.uint32)
  dtype = fs_gpu.dtype
  block_size = (16,16,1)
  grid_size = (1,int(np.ceil(float(basis_length)/block_size[1])))

  weights_gpu = cua.empty(basis_length, dtype=dtype)

  preproc = _generate_preproc(dtype, shape)
  preproc += '#define BLOCK_SIZE %d\n' % (block_size[0]*block_size[1])
  mod = SourceModule(preproc + projection_code, keep=True)

  projection_fun = mod.get_function("projection")

  projection_fun(weights_gpu.gpudata, fs_gpu.gpudata, basis_gpu.gpudata,
                 np.uint32(basis_length),
                 block=block_size, grid=grid_size)
  

def resize_stack_gpu(fs_gpu, out_shape):

  N = fs_gpu.shape[0]
  fs_in  = fs_gpu.get()
  del fs_gpu
  fs_out = np.zeros(tuple((np.int(N), np.int(out_shape[0]), np.int(out_shape[1]))))

  for i in range(N):
    fi_gpu = cua.to_gpu(fs_in[i])
    fi_gpu = resize_gpu(fi_gpu, out_shape)
    fs_out[i] = fi_gpu.get()

  fs_out = np.array(fs_out).astype(np.float32)

  return cua.to_gpu(fs_out)    
    
  
def resize_gpu(y_gpu, out_shape):

  in_shape = np.array(y_gpu.shape).astype(np.uint32)
  dtype = y_gpu.dtype
  if dtype != np.float32:
    raise NotImplementedException('Only float at the moment')
  block_size = (16,16,1)
  grid_size = (int(np.ceil(float(out_shape[1])/block_size[0])),
               int(np.ceil(float(out_shape[0])/block_size[1])))

  preproc = _generate_preproc(dtype)
  mod = SourceModule(preproc + resize_code, keep=True)

  resize_fun_gpu = mod.get_function("resize")
  resized_gpu = cua.empty(tuple((np.int(out_shape[0]),
                                 np.int(out_shape[1]))),y_gpu.dtype)

  temp_gpu, pitch = cu.mem_alloc_pitch(4 * y_gpu.shape[1],
                                       y_gpu.shape[0],
                                       4)
  copy_object = cu.Memcpy2D()
  copy_object.set_src_device(y_gpu.gpudata)
  copy_object.set_dst_device(temp_gpu)
  copy_object.src_pitch = 4 * y_gpu.shape[1]
  copy_object.dst_pitch = pitch
  copy_object.width_in_bytes = 4 * y_gpu.shape[1]
  copy_object.height = y_gpu.shape[0]
  copy_object(aligned=False)
  in_tex = mod.get_texref('in_tex')
  descr = cu.ArrayDescriptor()
  descr.width = y_gpu.shape[1]
  descr.height = y_gpu.shape[0]
  descr.format = cu.array_format.FLOAT
  descr.num_channels = 1
  #pitch = y_gpu.nbytes / y_gpu.shape[0]
  in_tex.set_address_2d(temp_gpu, descr, pitch)
  in_tex.set_filter_mode(cu.filter_mode.LINEAR)
  in_tex.set_flags(cu.TRSF_NORMALIZED_COORDINATES)
    
  resize_fun_gpu(resized_gpu.gpudata,
                 np.uint32(out_shape[0]), np.uint32(out_shape[1]),
                 block=block_size, grid=grid_size)
  temp_gpu.free()

  return resized_gpu


def rgb_to_yuv(image):

  trans_mat = np.array([[0.2126, 0.7152, 0.0722],
                        [-.1471,-0.2889, 0.4360],
                        [0.6150,-0.5150,-0.1000]])
  result = np.dot(image, trans_mat.T)
  return result


def rgb_to_yuv_gpu(y_gpu):
  raise NotImplementedError('Yet to be implemented')


def shock_filter_gpu(y_gpu, iter=3, dt=0.1, h=1):
  """Evolve image according to a "shock filter" process.

  This function does an in-place shock filtering on the input image.

  Args:
      y_gpu: Input image on GPU.
      iter: Number of time steps.
      dt: Duration of one time step.
      h: Size of grid steps.

  Returns:
      Nothing, but modifies y_gpu in place.
  """

  stream = cu.Stream()

  shape = np.array(y_gpu.shape).astype(np.uint32)
  dtype = y_gpu.dtype
  block_size = (16,16,1)
  grid_size = (int(np.ceil(float(shape[1])/block_size[0])),
               int(np.ceil(float(shape[0])/block_size[1])))
  shared_size = int((2+block_size[0])*(2+block_size[1])*dtype.itemsize)  

  preproc = _generate_preproc(dtype, shape)  
  mod = SourceModule(preproc + kernel_code, keep=True)

  shock_gpu = mod.get_function("shock")
  
  for i in range(iter):
    shock_gpu(y_gpu.gpudata, np.float32(dt), np.float32(h),
          block=block_size, grid=grid_size, stream=stream, shared=shared_size)
    stream.synchronize()
  

def sparsify_GPU(x_gpu, percentage):
  """
  Keeps only as many entries nonzero as specified by percentage.
  """
  x    = x_gpu.get()
  vals = np.sort(x.flatten())[::-1]
  idx  = np.floor(np.prod(x.shape) * percentage/100)
  cliplower_GPU(x_gpu, vals[idx])
  

def weighted_basis_gpu(psf_size, grid_size, im_size, params,
                       lens_file=None, lens_psf_size=None, lens_grid_size=None):

  # generate grid
  psf_size = psf_size + (1 - np.mod(psf_size, 2))
  
  if lens_file:
    grid = scipy.misc.imread(lens_file, flatten=True)
    if np.max(grid) > 255:
      grid /= 2**(16-1)
    else:
      grid /= 255
  else:
    grid = np.zeros(psf_size, dtype=np.float32)
    grid[(psf_size[0]-1)/2, (psf_size[1]-1)/2] = 1.
    grid = np.tile(grid, grid_size)
    lens_psf_size = psf_size
    #lens_grid_size = (1,1)
    lens_grid_size = grid_size

  grid_gpu = cu.matrix_to_array(grid, 'C')

  params_gpu = cu.matrix_to_array(params.astype(np.float32), 'C')

  block_size = (16,16,1)
  output_size = np.array(grid_size)*np.array(psf_size)
  gpu_grid_size = (int(np.ceil(float(output_size[1])/block_size[0])),
                   int(np.ceil(float(output_size[0])/block_size[1])))

  weighted_basis_gpu = cua.empty((int(output_size[0]),
                                  int(output_size[1])), np.float32)

  preproc = '' #_generate_preproc(basis_gpu.dtype)
  mod = SourceModule(preproc + basis_code, keep=True)
  basis_fun_gpu = mod.get_function("weighted_basis")

  in_tex = mod.get_texref('in_tex')
  in_tex.set_array(grid_gpu)
  in_tex.set_filter_mode(cu.filter_mode.LINEAR)
  #in_tex.set_flags(cu.TRSF_NORMALIZED_COORDINATES)

  params_tex = mod.get_texref('params_tex')
  params_tex.set_array(params_gpu)
  offset = ((np.array(im_size) - np.array(grid.shape)) /
            np.array(grid_size).astype(np.float32))
  offset = np.float32(offset)
  grid_scale = ((np.array(lens_grid_size) - 1) /
                (np.array(grid_size) - 1).astype(np.float32))
  grid_scale = np.float32(grid_scale)
  #psf_scale = ((np.array(lens_psf_size) - 1) /
  #             (np.array(psf_size) - 1).astype(np.float32))
  #psf_scale = np.float32(psf_scale)

  basis_fun_gpu(weighted_basis_gpu.gpudata,
                np.uint32(output_size[0]), np.uint32(output_size[1]),
                np.uint32(psf_size[0]), np.uint32(psf_size[1]),
                np.uint32(im_size[0]), np.uint32(im_size[1]),
                offset[0], offset[1],
                grid_scale[0], grid_scale[1],
                np.uint32(lens_psf_size[0]), np.uint32(lens_psf_size[1]),
                np.uint32(params.size/3),
                block=block_size, grid=gpu_grid_size)

  return weighted_basis_gpu
  

def wsparsify(w_gpu, percentage):
  """
  Keeps only as many entries nonzero as specified by percentage.
  """

  w    = w_gpu.get()
  vals = sort(w)[::-1]
  idx  = floor(prod(w.shape()) * percentage/100)
  zw_gpu = cua.zeros_like(w_gpu)   # gpu array filled with zeros
  tw_gpu = cua.empty_like(w_gpu)   # gpu array containing threshold
  tw_gpu.fill(vals[idx])        
  w_gpu  = cua.if_positive(w_gpu > tw_gpu, w_gpu, zw_gpu)

  del zw_gpu
  del tw_gpu

  return w_gpu


def yuv_to_rgb(image):

  trans_mat = np.array([[1., 0.1355, 1.3127],
                        [1.,-0.2591,-0.4077],
                        [1., 2.1676, 0.1729]]) 
  result = np.dot(image, trans_mat.T)
  return result


def yuv_to_rgb_gpu(image):
  raise NotImplementedError('Yet to be implemented')
    
if __name__ == "__main__":

    import pylab as pl
    import gputools
    import imagetools as it

    x = pl.imread('butcher.png')
    #x = np.random.rand(,1200).astype(np.float32)
    #x = it.rgb2gray(x)

    csf = (7,7)
    overlap = 0.5

    winaux = it.win2winaux(x.shape, csf, overlap)

    x_gpu = cua.to_gpu(x)
    xs_gpu = gputools.chop_pad_GPU(x_gpu, winaux.csf, winaux.sw, winaux.nhop, dtype='complex')
    start = time.clock()
    x_gpu = gputools.ola_GPU_test(xs_gpu, winaux.csf, winaux.sw, winaux.nhop)
    print "Time elapsed %.6f" % (time.clock()-start)


    x = np.real(x_gpu.get())

    #it.cellplot(xs,csf)
    pl.imshow(x)
    pl.show()

    #offset = (0,0)
    #sz = (xs.shape[1]-100, xs.shape[2]-100)

    #start = time.clock()   
    #xp_gpu = crop_stack_GPU(xs, sz, offset)
    #print "Time elapsed %.6f" % (time.clock()-start)
    #xp = np.real(xp_gpu.get())
    #xp = np.array(xp).astype(np.float32)

    #it.cellplot(xp,csf)
    #pl.show()

    #print "Computing FFT"
    #import pyfft.cuda as cufft
    #plan  = cufft.Plan((xs_gpu.shape[1],xs_gpu.shape[2]))

    #start = time.clock()       
    #plan.execute(xs_gpu,batch=xs_gpu.shape[0])
    #plan.execute(xs_gpu,batch=xs_gpu.shape[0],inverse=True)
    #print "Time elapsed %.6f" % (time.clock()-start)


    #from pylab import *
    #from imagetools import *
    #from cnvtools import crop, pad
    # 
    #doplot = 1
    # 
    #print '---------------------------------------'
    #print 'Test cropping kernel'
    #print '---------------------------------------'
    # 
    #x     = np.array(imread('butcher.png'))
    #x_cpx = x.astype(np.complex64)
    #x_gpu = cua.to_gpu(x_cpx)
    # 
    #sfft   = (272,126)
    #offset = (269,529)
    # 
    #start = time.clock()
    #xc_gpu = crop_gpu2cpu(x_gpu, sfft, offset)
    #print 'GPU cropping %.6f' % (time.clock()-start)
    # 
    #start = time.clock()
    #x_cpu  = np.array(real(x_gpu.get()))
    #xc_cpu = crop(x_cpu, sfft, offset)
    #print 'CPU cropping %.6f' % (time.clock()-start)
    # 
    #print '---------------------------------------'
    #print 'Difference for cropping kernel %.6f' % np.sum((xc_cpu-xc_gpu).flatten())
    #print '---------------------------------------'
    #print ''
    # 
    #if doplot:
    #    figure(1)
    #    title('Cropping kernel')
    #    subplot(131)
    #    imshow(xc_cpu)
    #    title('CPU')
    #    subplot(132)
    #    imshow(xc_gpu)
    #    title('GPU')
    #    subplot(133)
    #    imshow(xc_cpu-xc_gpu)
    #    title('Difference image')
    #    draw()
    #    
    #    
    #print ''
    #print '---------------------------------------'
    #print 'Test padding kernel'
    #print '---------------------------------------'
    # 
    # 
    #sfft   = (2048,2048)
    #offset = (61,61)
    # 
    #start  = time.clock()
    #xp_gpu = pad_cpu2gpu(x, sfft, offset, dtype='complex')
    #print 'GPU padding %.6f' % (time.clock()-start)     
    #xp_gpu = xp_gpu.get()
    # 
    #start = time.clock()
    #xp    = (pad(x, sfft, offset)).astype(np.complex64)
    #x_cpu = cua.to_gpu(xp)
    #print 'CPU padding %.6f' % (time.clock()-start)
    #xp_cpu = x_cpu.get()
    # 
    #print '---------------------------------------'
    #print 'Difference for padding kernel %.6f' % np.sum((xp_cpu-xp_gpu).flatten())
    #print '---------------------------------------'
    #print ''
    # 
    #if doplot:
    #    figure(2)
    #    subplot(131)
    #    imshow(real(xp_cpu))
    #    title('CPU')
    #    subplot(132)
    #    imshow(real(xp_gpu))
    #    title('GPU')
    #    subplot(133)
    #    imshow(real(xp_cpu-xp_gpu))
    #    title('Difference image')
    #    draw()

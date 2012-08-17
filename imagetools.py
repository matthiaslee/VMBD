# Import other libraries 
import pycuda.gpuarray as cua
import matplotlib.pyplot as mp
import numpy as np
import pylab
from scipy import signal
from PIL import Image

# Import own libraries 
import gputools


def pad(x, sz, offset=0, value=0.):
    """
    Pads a n-dimensional array 'x' to size 'sz' with a real number specified
    by 'value'. With 'offset' an additional offset can be specified.

    --------------------------------------------------------------------------
    Usage:
    
    Call:  y = pad(x, sz, offset=0, value=0)
    
    Input:  x         n-dimensional numpy array              
            sz        size of padded  output                 
            offset    optional offset (0 by default)         
            value     value of padding (0 by default)        

    Output: z         padded n-dimensional array of size sz
    --------------------------------------------------------------------------

    Copyright (C) 2011 Michael Hirsch   
    """
    from numpy import array, ones

    offset = offset * ones(x.ndim)
    sx = array(x.shape)
    sz = array(sz)
    z  = value * ones(sz)
    slices    = [slice(offset[i],offset[i]+sx[i]) for i in range(len(sz))]
    z[slices] = x
    
    return z

def crop(x, sz, offset=0):
    """
    Crops a n-dimensional array 'x' to size 'sz'.
    With 'offset' an additional offset can be specified.

    --------------------------------------------------------------------------
    Usage:
    
    Call:  y = crop(x, sz, offset=0)
    
    Input:  x         n-dimensional numpy array              
            sz        size of padded  output, sz > shape(x)!                
            offset    optional offset (0 by default)         

    Output: z         padded n-dimensional array of size sz
    --------------------------------------------------------------------------

    Copyright (C) 2011 Michael Hirsch   
    """
    from numpy import ones, array

    sz = array(sz)
    offset = offset * ones(x.ndim)  
    slices = [slice(offset[i],offset[i]+sz[i]) for i in range(len(sz))]
    z = x[slices]
    
    return z


def circshift(X, T):
    """    
    Does shift with circular boundary conditions

    --------------------------------------------------------------------------
    Usage:
    
    Call:  y = circshift(x, T)
    
    Input: x   image
           T   two-dimensional translation vector

    Output: Image y shifted with circular boundary conditions
    --------------------------------------------------------------------------

    Copyright (C) 2011 Michael Hirsch   
    """
     
    t = T % np.array(X.shape)
    
    X = np.concatenate((X[-t[0]:], X[:-t[0]]),0)
    X = np.concatenate((X[:,-t[1]:], X[:,:-t[1]]),1)
    
    return X       


def randomwalk(sf, N=None):
    """
    Generates a randomwalk in a ndarray of given shape sf and length N

    --------------------------------------------------------------------------
    Usage:
    
    Call:  w = randomwalk(sf, N=None)
    
    Input: sf  size of ndarray w
           N   length of randomwalk

    Output: ndarray w containing randomwalk of length N.
            Note, that w is normalized to w.sum() = 1
    --------------------------------------------------------------------------

    Copyright (C) 2011 Michael Hirsch   
    """
    if N == None:
        N = np.floor(np.prod(sf)/100)
        
    w = np.zeros(sf);
    ndims   = len(np.shape(w))
    center    = np.ceil(np.array(sf)/2)
    w[tuple(center)] = 1.

    loc = center
    for i in np.arange(N):
        loc += ((pylab.rand(ndims)-0.5)*2).round()
        loc = clip(loc,1,np.array(sf).min()-1)
        w[tuple(loc)] += 1

    w /= w.sum()
    return w


def clip(x, lb=0, ub=1e300):
    """
    Clips an ndarray at a lower bound lb and a upper bound ub

    --------------------------------------------------------------------------
    Usage:
    
    Call:  y = clip(x, lb=0, ub=1e300)
    
    Input: x   input ndarray x
           lb  lower bound
           ub  upper bound

    Output: clipped ndarray of same size as x            
    --------------------------------------------------------------------------

    Copyright (C) 2011 Michael Hirsch   
     """    
    return x.clip(lb,ub)

    
def plotcube(x):
    """
    Plots projections of a 3d ndarray

    --------------------------------------------------------------------------
    Usage:
    
    Call:  plotcube(x)
    
    Input: x   input ndarray x

    Output: Shows projections of x            
    --------------------------------------------------------------------------

    Copyright (C) 2011 Michael Hirsch   
    """
    mp.subplot(131)
    mp.imshow(np.sum(x,0))
    mp.title('Projection in x')
    mp.subplot(132)
    mp.imshow(np.sum(x,1))
    mp.title('Projection in y')
    mp.subplot(133)
    mp.imshow(np.sum(x,2))
    mp.title('Projection in z')
    mp.draw()
    mp.show()


def cellplot(fs, csf):
    """
    Plots PSF kernels
    
    --------------------------------------------------------------------------
    Usage:
    
    Call:  cellplot(fs, csf)
    
    Input: fs   PSF kernels, i.e. 3d array with kernels indexed by 0th index
           csf  size of kernels in x and y direction
 
    Output: Shows stack of PSF kernels arranged according to csf
    --------------------------------------------------------------------------

    Copyright (C) 2011 Michael Hirsch
    """    

    mp.clf()
    for i in range(np.prod(csf)):
        mp.subplot(csf[0],csf[1],i+1)
        mp.imshow(fs[i])
        mp.axis('off')
    mp.draw()


def gridF(fs, csf):
    """
    Concatenates PSF kernels to one 2d image, potentially useful for plotting.
    
    --------------------------------------------------------------------------
    Usage:
    
    Call:  gridF(fs, csf)
    
    Input: fs   PSF kernels, i.e. 3d array with kernels indexed by 0th index
           csf  size of kernels in x and y direction
 
    Output: 2d image with PSF kernels arranged according to csf
    --------------------------------------------------------------------------

    Copyright (C) 2011 Michael Hirsch
    """    

    f = np.ones((fs.shape[1]*csf[0],fs.shape[2]*csf[1]))

    for i in range(np.prod(csf)):
        k = i/csf[0]
        l = np.remainder(i,csf[0])
        
        f[k * fs.shape[1]:(k+1)*fs.shape[1],
          l * fs.shape[2]:(l+1)*fs.shape[2]] = fs[i,:,:]

    return f


def sparsify(x, percentage):
    """
    Keeps only as many entries nonzero as specified by percentage.
    Note that only the larges values are kept.
  
    --------------------------------------------------------------------------
    Usage:
    
    Call:  y = sparsify(x, percentage)
    
    Input: x            input ndarray x
           percentage   percentage of nonzero entries in y 

    Output: sparsified version of x            
    --------------------------------------------------------------------------

    Copyright (C) 2011 Michael Hirsch   
    """
    vals = np.sort(x.flatten())[::-1]
    idx  = np.floor(np.prod(x.shape) * percentage/100)
    x[x < vals[idx]] = 0
  
    return x


def rgb2gray(x):
    """
    Transforms an image from RGB color space to grayscale
    by taking the mean over its color channels.
  
    --------------------------------------------------------------------------
    Usage:
    
    Call:  y = rgb2gray(x)
    
    Input: x    input image x

    Output: y   for now green color channel only            
    --------------------------------------------------------------------------

    Copyright (C) 2010 Michael Hirsch
    """
    
    if len(x.shape) > 2:
        x = x[:,:,1]
        
    return np.array(x)


def imwrite(x, filename):
    """
    Write images as 8bit images to disk. Convenient wrapper around save
    function of PIL library
    
    --------------------------------------------------------------------------
    Usage:
    
    Call:  imwrite(x, filename)
    
    Input: x          input image x
           filename   string where to save image           
    --------------------------------------------------------------------------

    Copyright (C) 2010 Michael Hirsch
    """    
    # lets not muck around with the original
    t_x = x.copy()
    if not t_x.dtype == 'uint8':
        t_x *= 255.
    imx = Image.fromarray(np.uint8(t_x))
    imx.save(filename)
    

def window(sw, window = 'barthann'):
    """
    Creates 2d window of size sw
    
    --------------------------------------------------------------------------
    Usage:
    
    Call:  w = window(sw, window = 'barthann')
    
    Input: sw       size of window 
           window   string specifying window type

    Output: Window of size sw 
    --------------------------------------------------------------------------

    Copyright (C) 2010 Michael Hirsch
    """    

    w1  = signal.get_window(window,sw[0])
    w1  = (w1 + w1[::-1])/2
    w1 -= w1.min()
    w2  = signal.get_window(window,sw[1])
    w2  = (w2 + w2[::-1])/2
    w2 -= w2.min()

    www = np.outer(w1,w2)
    www = www/www.max()

    www = np.maximum(www, 1e-16)

    return www


def edgetaper(x, sf, windowfun='barthann'):
    """
    Applies edgetapering of size sf to image x with specified window type.
    Note that x can be a 3d patch stack as well. In this case each patch is 
    edgetapered.
    
    --------------------------------------------------------------------------
    Usage:
    
    Call:  y = edgetaper(x, sf, win='barthann')
    
    Input: x          input image
           sf         size of edgetapering in affected pixels on both sides,
                      i.e. sf/2 pixels on one side is affected by edgetapering
           windowfun  window type 

    Output: Edgetapered image
    --------------------------------------------------------------------------

    Copyright (C) 2011 Michael Hirsch
    """    

    if len(x.shape) == 2:
        sx = x.shape
    elif len(x.shape) == 3:        
        sx = x.shape[1:]
    else:
        raise error("[imagetools.py] edgetaper")
        
    # Ensure that sf is odd
    sf = sf+(1-np.mod(sf,2))
    dsx = sx - sf

    # Create window
    w = window(sf, windowfun)

    # Split into halves, tile and concatenate again
    wins = np.array_split(w,2,axis = 1)
    w = np.hstack([wins[0],np.tile(wins[0][:,-1],(dsx[1],1)).T,wins[1]])

    wins = np.array_split(w,2,axis = 0)
    w = np.vstack([wins[0],np.tile(wins[0][-1,:],(dsx[0],1)),wins[1]])

    w = w/w.max()

    if len(x.shape) == 3:
        w = np.tile(w,(x.shape[0],1,1))
        
    return x*w


class win2winaux:
    """
    Creates an instance containing all necessary information for being passed
    to OlaGPU. In particular, it gives back a stack of windows, one for each
    patch of information for overlap-and-add. Note that the windows are a
    already transferred to GPU
        
    --------------------------------------------------------------------------
    Usage:
    
    Call:  winaux = win2winaux(sx, csf, overlap, windowfun='barthann')
    
    Input: sx         total size of image
           csf        number of PSF kernels in x and y direction
           overlap    overlap of neighboring windows in percent, e.g. 0.5
           windowfun  window type 

    Output: Instance containing stack of windows stored on GPU and all
            necessary information for calling OlaGPU
    --------------------------------------------------------------------------

    Copyright (C) 2011 Michael Hirsch
    """    

    def __init__(self, sx, csf, overlap, windowfun='barthann'):
        hop    = 1 - overlap
        factor = np.array(csf) * hop + overlap
        sw     = np.ceil(sx / factor)
    
        swold = sw;
        
        while any(np.floor(sw * overlap) != sw * overlap):
            sw = sw + 1;
            if any(sw > 2 * swold):
                sw = swold
                break
            
        noverlap = np.floor(sw * overlap) 
        nhop = sw - noverlap

        w  = window(sw, window = 'barthann')
        ws = np.tile(w, (np.prod(csf),1,1))
        ws = np.array(ws).astype(np.float32)
        nom_gpu   = cua.to_gpu(ws)
        w_gpu     = gputools.ola_GPU_test(nom_gpu, csf, sw, nhop)
        denom_gpu = gputools.chop_pad_GPU(w_gpu, csf, sw, nhop, sw, dtype='real')
        ws_gpu  = cua.empty(nom_gpu.shape,np.float32)
        ws_gpu  = nom_gpu / denom_gpu
        
        self.sx   = sx
        self.sw   = sw
        self.csf  = csf
        self.nhop = nhop
        self.ws_gpu = ws_gpu
        self.nom_gpu = nom_gpu
        self.denom_gpu = denom_gpu
        self.w_gpu = w_gpu
                
                
                





        
    

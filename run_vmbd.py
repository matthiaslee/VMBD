#!/usr/bin/env python
# Script for performing spatially-varying online multi-frame
# blind deconvolution
#
# Copyright (C) 2011 Michael Hirsch   


# Load other libraries
import pycuda.autoinit
import pycuda.gpuarray as cua
import numpy as np
# pylab is turned off because it takes forever to load
#import pylab as pl
#from pylab import draw, figure, bla bla bla
import os
from shutil import rmtree 
from optparse import OptionParser

# Load own libraris
import olaGPU 
import imagetools
import gputools
import fitsTools
import stopwatch
import img_scale

def process(opts):
    # ============================================================================
    # Specify some parameter settings 
    # ----------------------------------------------------------------------------
    
    # Specify data path and file identifier
    DATAPATH = '/DATA/LSST/FITS'
    RESPATH  = '../../../DATA/results';
    BASE_N = 141
    #FILENAME = lambda i: '%s/v88827%03d-fz.R22.S11.fits.png' % (DATAPATH,(BASE_N+i))
    FILENAME = lambda i: '%s/v88827%03d-fz.R22.S11.fits' % (DATAPATH,(BASE_N+i))
    ID       = 'LSST'
    
    # ----------------------------------------------------------------------------
    # Specify parameter settings
    
    # General
    doshow   = opts.doShow             # put 1 to show intermediate results
    backup   = opts.backup                  # put 1 to write intermediate results to disk
    N        = opts.N                # how many frames to process
    N0       = opts.N0                # number of averaged frames for initialisation
    
    # OlaGPU parameters
    sf      = np.array([40,40])   # estimated size of PSF
    csf     =(3,3)               # number of kernels across x and y direction
    overlap = 0.5                 # overlap of neighboring patches in percent
    
    # Regularization parameters for kernel estimation
    f_alpha = opts.f_alpha                 # promotes smoothness
    f_beta  = opts.f_beta         # Thikhonov regularization
    optiter = opts.optiter        # number of iterations for minimization
    tol     = opts.tol        # tolerance for when to stop minimization
    # ============================================================================
    
    # Create helper functions for file handling
    
    # # # HACK for chunking into available GPU mem # # #
    #     - loads one 1kx1k block out of the fits image
    xOffset=2000
    yOffset=0
    chunkSize=1000
    yload = lambda i: 1. * fitsTools.readFITS(FILENAME(i), use_mask=True, norm=True)[yOffset:yOffset+chunkSize,xOffset:xOffset+chunkSize]
    
    # ----------------------------------------------------------------------------
    # Some more code for backuping the results
    # ----------------------------------------------------------------------------
    # For backup purposes
    EXPPATH = '%s/%s_sf%dx%d_csf%dx%d_maxiter%d_alpha%.2f_beta%.2f' % \
          (RESPATH,ID,sf[0],sf[1],csf[0],csf[1],optiter,f_alpha,f_beta)
          
    xname = lambda i: '%s/x_%04d.png' % (EXPPATH,i)
    yname = lambda i: '%s/y_%04d.png' % (EXPPATH,i)
    fname = lambda i: '%s/f_%04d.png' % (EXPPATH,i)
    
    
    if os.path.exists(EXPPATH) and opts.overwrite:
        try:
            rmtree(EXPPATH)
        except:
            print "[ERROR] removing old results dir:",EXPPATH
            exit()
            
    elif os.path.exists(EXPPATH):
        print "[ERROR] results directory already exists, please remove or use '-o' to overwrite"
        exit()
        
    # Create results path if not existing
    try:
        os.makedirs(EXPPATH)
    except:
        print "[ERROR] creating results dir:",EXPPATH
        exit()
    
    print 'Results are saved to: \n %s \n' % EXPPATH
    
    # ----------------------------------------------------------------------------
    # For displaying intermediate results create target figure
    # ----------------------------------------------------------------------------
    # Create figure for displaying intermediate results
    if doshow:
        print "showing intermediate results is currently disabled.."
        #pl.figure(1)
        #pl.draw()
    
    # ----------------------------------------------------------------------------
    # Code for initialising the online multi-frame deconvolution
    # ----------------------------------------------------------------------------
    # Initialisation of latent image by averaging the first 20 frames
    y0 = 0.
    for i in np.arange(1,N0):
        y0 += yload(i)
    
    y0 /= N0
    y_gpu = cua.to_gpu(y0)
    
    # Pad image since we perform deconvolution with valid boundary conditions
    x_gpu = gputools.impad_gpu(y_gpu, sf-1)
    
    # Create windows for OlaGPU
    sx      = y0.shape + sf - 1
    sf2     = np.floor(sf/2)
    winaux  = imagetools.win2winaux(sx, csf, overlap)
    
    # ----------------------------------------------------------------------------
    # Loop over all frames and do online blind deconvolution
    # ----------------------------------------------------------------------------
    import time as t
    ti = t.clock()
    t1 = stopwatch.timer()
    t2 = stopwatch.timer()
    t3 = stopwatch.timer()
    t4 = stopwatch.timer()
    t4.start()
    for i in np.arange(1,N+1):
        print 'Processing frame %d/%d \r' % (i,N)
    
        # Load next observed image
        t3.start()
        y = yload(i)
        print "TIMER load:", t3.elapsed()
        
        # Compute mask for determining saturated regions
        mask_gpu = 1. * cua.to_gpu(y < 1.)
        y_gpu    = cua.to_gpu(y)
    
        # ------------------------------------------------------------------------
        # PSF estimation
        # ------------------------------------------------------------------------
        # Create OlaGPU instance with current estimate of latent image
        t2.start()
        X = olaGPU.OlaGPU(x_gpu,sf,'valid',winaux=winaux)
        print "TIMER GPU: ", t2.elapsed()
        
        t1.start()
        # PSF estimation for given estimate of latent image and current observation
        f = X.deconv(y_gpu, mode = 'lbfgsb', alpha = f_alpha, beta = f_beta,
             maxfun = optiter, verbose = 10)
        print "TIMER Optimization: ", t1.elapsed()
        #print "F: ",type(f),f
        fs = f[0]
    
        # Normalize PSF kernels to sum up to one
        fs = gputools.normalize(fs)
    
        # ------------------------------------------------------------------------
        # Latent image estimation
        # ------------------------------------------------------------------------
        # Create OlaGPU instance with estimated PSF
        t2.start()
        F = olaGPU.OlaGPU(fs,sx,'valid',winaux=winaux)
    
        # Latent image estimation by performing one gradient descent step
        # multiplicative update is used which preserves positivity 
        factor_gpu = F.cnvtp(mask_gpu*y_gpu)/(F.cnvtp(mask_gpu*F.cnv(x_gpu))+tol)
        gputools.cliplower_GPU(factor_gpu, tol)
        x_gpu = x_gpu * factor_gpu
        x_max = x_gpu.get()[sf[0]:-sf[0],sf[1]:-sf[1]].max()
        
        gputools.clipupper_GPU(x_gpu, x_max)
        print "TIMER GPU: ", t2.elapsed()
        
        # ------------------------------------------------------------------------
        # For backup intermediate results
        # ------------------------------------------------------------------------
        if backup or i == N:
            # Write intermediate results to disk incl. input
            y_img = y_gpu.get()*1e5
            #print "y",y_img.max(), type(y)
            #print y_img.shape
            #print y_img
            #fitsTools.fitsStats(y_img)
            fitsTools.asinhScale(y_img, 450, -50, minCut=0.0, maxCut=40000, fname=yname(i))
            #imagetools.imwrite(y_img, yname(i))
            
            # Crop image to input size
            xi = (x_gpu.get()[sf2[0]:-sf2[0],sf2[1]:-sf2[1]] / x_max)*1e5
            #print "xi",  xi.min(), xi.max(), type(xi)
            #print xi.shape
            #print xi
            fitsTools.fitsStats(xi)
            fitsTools.asinhScale(xi, 450, -50, minCut=0.0, maxCut=40000, fname=xname(i))
            #imagetools.imwrite(xi, xname(i))
        
            # Concatenate PSF kernels for ease of visualisation
            f = imagetools.gridF(fs,csf)
            f = f*1e5
            
            fitsTools.asinhScale(f, 450, -50, minCut=0.0, maxCut=40000, fname=fname(i))
            #imagetools.imwrite(f/f.max(), fname(i))
            #exit()
    
        # ------------------------------------------------------------------------
        # For displaying intermediate results
        # ------------------------------------------------------------------------
        '''
        if np.mod(i,1) == 0 and doshow:
        pl.figure(1)
        pl.subplot(121)
        # what is SY?
        pl.imshow(imagetools.crop(x_gpu.get(),sy,np.ceil(sf/2)),'gray')
        pl.title('x after %d observations' % i)
        pl.subplot(122)
        pl.imshow(y_gpu.get(),'gray')
        pl.title('y(%d)' % i)
        pl.draw()
        pl.figure(2)
        pl.title('PSF(%d)' % i)
        imagetools.cellplot(fs, winaux.csf)
        tf = t.clock()
        print('Time elapsed after %d frames %.3f' % (i,(tf-ti)))
        '''
    tf = t.clock()
    print('Time elapsed for total image sequence %.3f' % (tf-ti))
    # ----------------------------------------------------------------------------
    print "TOTAL: %.3f" % (t4.elapsed())
    print "OptimizeCPUtime %.3f %.3f" % (t1.getTotal(), 100*(t1.getTotal()/t4.getTotal()))
    print "GPUtime %.3f %.3f" % (t2.getTotal(), 100*(t2.getTotal()/t4.getTotal()))
    print "LoadTime %.3f %.3f" % (t3.getTotal(), 100*(t3.getTotal()/t4.getTotal()))
    

if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option("-s","--doShow", action="store_true", dest="doShow", default=False, help="show output at every timestep (Default: False)")
    optparser.add_option("-b","--backup", action="store_true", dest="backup", default=False, help="write intermediate results to disk (Default: False)")
    optparser.add_option("-l","--log", action="store_true", dest="log", default=False, help="write log to file (Default: False)")
    optparser.add_option("-o","--overwrite", action="store_true", dest="overwrite", default=False, help="overwrite existing results (Default: False)")
    optparser.add_option("-n","--nFrames", dest="N", default=100, type="int", help="Number of frames to process (Default: 100)")
    optparser.add_option("-a","--nAveFrames", dest="N0", default=20, type="int", help="Number of frames averaged for initialization (Default: 20)")
    optparser.add_option("--f_alpha", dest="f_alpha", default=0.0, type="float", help="promotes smoothness (Default: 0.0)")
    optparser.add_option("--f_beta", dest="f_beta", default=0.1, type="float", help="Thikhonov regularization (Default: 0.1)")
    optparser.add_option("--optiter", dest="optiter", default=50, type="float", help="number of iterations (Default: 50)")
    optparser.add_option("--tol", dest="tol", default=1e-10, type="float", help="tolerance for when to stop minimization (Default: 1e-10)")
    
    (opts,args) = optparser.parse_args()
    
    print "Set Parameters:", opts, args, "\n"
    process(opts)
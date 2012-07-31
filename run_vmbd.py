#!/usr/bin/env python
# Script for performing spatially-varying online multi-frame
# blind deconvolution
#
# Copyright (C) 2011 Michael Hirsch   


# Load other libraries
import pycuda.autoinit
import pycuda.gpuarray as cua
import numpy as np
import pylab as pl
import os
import optparse

# Load own libraris
import olaGPU 
import imagetools
import gputools
import fitsTools

def process():
	# ============================================================================
	# Specify some parameter settings 
	# ----------------------------------------------------------------------------
	
	# Specify data path and file identifier
	DATAPATH = '/home/madmaze/DATA/LSST/ds9/gPNG'
	DATAPATH2 = '/home/madmaze/DATA/LSST/FITS'
	RESPATH  = '/home/madmaze/DATA/LSST/results';
	BASE_N = 141
	FILENAME = lambda i: '%s/v88827%03d-fz.R22.S11.fits.png' % (DATAPATH,(BASE_N+i))
	FILENAME2 = lambda i: '%s/v88827%03d-fz.R22.S11.fits' % (DATAPATH2,(BASE_N+i))
	ID       = 'LSST'
	
	# ----------------------------------------------------------------------------
	# Specify parameter settings
	
	# General
	doshow   = 0                  # put 1 to show intermediate results
	backup   = 1                  # put 1 to write intermediate results to disk
	N        = 20 #100                # how many frames to process
	N0       = 10 #20                # number of averaged frames for initialisation
	
	# OlaGPU parameters
	sf      = np.array([20,20])   # estimated size of PSF
	csf     = (3,3)               # number of kernels across x and y direction
	overlap = 0.5                 # overlab of neighboring patches in percent
	
	# Regularization parameters for kernel estimation
	f_alpha = 0.                 # promotes smoothness
	f_beta  = 0.1                  # Thikhonov regularization
	optiter = 50                 # number of iterations for minimization
	tol     = 1e-10               # tolerance for when to stop minimization
	# ============================================================================
	
	
	
	# Hopefully no need to edit anything below
	
	# ----------------------------------------------------------------------------
	# Some more code for backuping the results
	# ----------------------------------------------------------------------------
	if backup:
	    # Create helper functions for file handling
	    # yload = lambda i: 1. * pl.imread(FILENAME(i)).astype(np.float32)
	    xOffset=2000
	    yOffset=0
	    chunkSize=1000
	    yload = lambda i: 1. * fitsTools.readFITS(FILENAME2(i))[yOffset:yOffset+chunkSize,xOffset:xOffset+chunkSize].astype(np.float32)
	    
	
	    # For backup purposes
	    EXPPATH = '%s/%s_sf%dx%d_csf%dx%d_maxiter%d_alpha%.2f_beta%.2f' % \
		      (RESPATH,ID,sf[0],sf[1],csf[0],csf[1],optiter,f_alpha,f_beta)
	
	    xname = lambda i: '%s/x_%04d.png' % (EXPPATH,i)
	    yname = lambda i: '%s/y_%04d.png' % (EXPPATH,i)
	    fname = lambda i: '%s/f_%04d.png' % (EXPPATH,i)
	
	    # Create results path if not existing
	    try:
		os.makedirs(EXPPATH)
	    except:
		pass
	
	    print 'Results are saved to: \n %s \n' % EXPPATH
	
	# ----------------------------------------------------------------------------
	# For displaying intermediate results create target figure
	# ----------------------------------------------------------------------------
	# Create figure for displaying intermediate results
	if doshow:
	    pl.figure(1)
	    pl.draw()
	
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
	
	for i in np.arange(1,N+1):
	
	    print 'Processing frame %d/%d \r' % (i,N)
	
	    # Load next observed image
	    y = yload(i)
	    
	    # Compute mask for determining saturated regions
	    mask_gpu = 1. * cua.to_gpu(y < 1.)
	    y_gpu    = cua.to_gpu(y)
	
	    # ------------------------------------------------------------------------
	    # PSF estimation
	    # ------------------------------------------------------------------------
	    # Create OlaGPU instance with current estimate of latent image
	    X = olaGPU.OlaGPU(x_gpu,sf,'valid',winaux=winaux)
	
	    # PSF estimation for given estimate of latent image and current observation
	    f = X.deconv(y_gpu, mode = 'lbfgsb', alpha = f_alpha, beta = f_beta,
			 maxfun = optiter, verbose = 10)
	    #print "F: ",type(f),f
	    fs = f[0]
	
	    # Normalize PSF kernels to sum up to one
	    fs = gputools.normalize(fs)
	
	    # ------------------------------------------------------------------------
	    # Latent image estimation
	    # ------------------------------------------------------------------------
	    # Create OlaGPU instance with estimated PSF
	    F = olaGPU.OlaGPU(fs,sx,'valid',winaux=winaux)
	
	    # Latent image estimation by performing one gradient descent step
	    # multiplicative update is used which preserves positivity 
	    factor_gpu = F.cnvtp(mask_gpu*y_gpu)/(F.cnvtp(mask_gpu*F.cnv(x_gpu))+tol)
	    gputools.cliplower_GPU(factor_gpu, tol)
	    x_gpu = x_gpu * factor_gpu
	    x_max = x_gpu.get()[sf[0]:-sf[0],sf[1]:-sf[1]].max()
	    gputools.clipupper_GPU(x_gpu, x_max)
	
	    
	    # ------------------------------------------------------------------------
	    # For backup intermediate results
	    # ------------------------------------------------------------------------
	    if backup:
		# Write intermediate results to disk incl. input
		 imagetools.imwrite(y_gpu.get(), yname(i))
	
		 # Crop image to input size
		 xi = x_gpu.get()[sf2[0]:-sf2[0],sf2[1]:-sf2[1]] / x_max
		 imagetools.imwrite(xi, xname(i))
	
		 # Concatenate PSF kernels for ease of visualisation
		 f = imagetools.gridF(fs,csf)
		 imagetools.imwrite(f/f.max(), fname(i))
	
	
	    # ------------------------------------------------------------------------
	    # For displaying intermediate results
	    # ------------------------------------------------------------------------
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
	
	tf = t.clock()
	print('Time elapsed for total image sequence %.3f' % (tf-ti))
	# ----------------------------------------------------------------------------

if __name__ == "__main__":
	optparser = optparse.OptionParser()
	optparser.add_option("-s","--doShow", action="store_true", dest="doShow", default=False, help="show output at every timestep (Default: False)")
	optparser.add_option("-b","--backup", action="store_true", dest="backup", default=False, help="write intermediate results to disk (Default: False)")
	optparser.add_option("-l","--log", action="store_true", dest="log", default=False, help="write log to file (Default: False)")
	optparser.add_option("-n","--nFrames", dest="N", default=100, type="int", help="Number of frames to process (Default: 100)")
	optparser.add_option("-a","--nAveFrames", dest="N0", default=20, type="int", help="Number of frames averaged for initialization (Default: 20)")
	optparser.add_option("--f_alpha", dest="f_alpha", default=0.0, type="float", help="promotes smoothness (Default: 0.0)")
	optparser.add_option("--f_beta", dest="f_beta", default=0.1, type="float", help="Thikhonov regularization (Default: 0.1)")
	optparser.add_option("--optiter", dest="optiter", default=50, type="float", help="number of iterations (Default: 50)")
	optparser.add_option("--tol", dest="tol", default=1e-10, type="float", help="tolerance for when to stop minimization (Default: 1e-10)")
	
	(opts,args) = optparser.parse_args()
	
	print "Set Parameters:", opts, args
	
	process(opts[:])

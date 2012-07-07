#!/usr/bin/env python

import pyfits
import os
import types
import pylab
import numpy
import img_scale

def readFITS(fitsPath):
	print "loading %s..." % fitsPath
	hdulist = pyfits.open(fitsPath)
	n=0
	data=[]
	
	'''
	while n < len(hdulist):
		if type(hdulist[n].data) != types.NoneType:
			print "Found hdulist[%d].data of type: %s" % (n, type(hdulist[n].data))
			data.append(hdulist[n].data)
		else:
			print "nothing at hdulist[%d].data" % (n) 
		n+=1
	'''
	# get first element
	raw_img_data = hdulist[1].data
	hdulist.close()
	width=raw_img_data.shape[0]
	height=raw_img_data.shape[1]
	# cast into float
	raw_img_data = numpy.array(raw_img_data, dtype=float)
	sig_fract = 5.0
	percent_fract = 0.05
	#sky, num_iter = img_scale.sky_mean_sig_clip(raw_img_data, sig_fract, percent_fract, max_iter=10)
	#print "sky = ", sky, '(', num_iter, ')'
	img_data =  raw_img_data#-sky#+500
	tmp_array = numpy.array(raw_img_data, copy=True)
	idx = numpy.where(tmp_array < 0.0)
	tmp_array[idx] = 0.0
	new_img = img_scale.asinh(img_data, scale_min=0.0,non_linear=300.0)
	#new_img = img_scale.sqrt(img_data, scale_min=-1.0)
	return new_img

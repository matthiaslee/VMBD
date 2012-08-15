#!/usr/bin/env python

import pyfits
import os
import types
import pylab
import numpy
import img_scale
import stopwatch
import imagetools

def readFITS(fitsPath):
	t1 = stopwatch.timer()
	
	t1.start()
	print "loading %s..." % fitsPath
	hdulist = pyfits.open(fitsPath)
	print "load:",t1.elapsed()
	n=0
	data=[]

	t1.start()
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
	#tmp_array = numpy.array(raw_img_data, copy=True)
	#idx = numpy.where(tmp_array < 0.0)
	#tmp_array[idx] = 0.0
	print "preproc:",t1.elapsed()
	t1.start()
	#new_img = img_scale.asinh(img_data, scale_min=0.0,non_linear=300.0)
	print "scale:",t1.elapsed()
	#new_img = img_scale.sqrt(img_data, scale_min=-1.0)
	return numpy.array(raw_img_data, copy=True)
	
	#return new_img
def asinhScale(data, nonlin, shift, minCut, maxCut, fname="", rgb=False):
	output = numpy.array(data, copy=True)
	
	fact=numpy.arcsinh((maxCut-minCut)/nonlin)
	minX=data.min()
	maxX=data.max()
	
	output = output + shift
	
	lowCut = numpy.where(output < minCut)
	data_i = numpy.where((output > minCut) & (output < maxCut))
	hiCut = numpy.where(output > maxCut)
	
	output[lowCut] = 0.0
	output[data_i] = numpy.arcsinh(((output[data_i])/nonlin))/fact
	output[hiCut] = 1.0
	
	#plt.plot(dat.flatten(),output.flatten(),label=(str(nonlin)+'/'+str(shift)+'/'+str(minCut)+'/'+str(maxCut)))
	#plt.legend(loc=4)
	#plt.show()
	if fname != "":
		imagetools.imwrite(output, fname+(str(nonlin)+'-'+str(shift)+'-'+str(minCut)+'-'+str(maxCut))+".png")
	if rgb is True:
		rgbImg = numpy.zeros((output.shape[0], output.shape[1], 3), dtype=float)
		#print output[lowCut].shape
		print len(lowCut[0])
		# make low end visible
		#output[lowCut] = 1.0
		rgbImg[data_i[0],data_i[1],0] = output[data_i]
		rgbImg[data_i[0],data_i[1],2] = output[data_i]
		
		rgbImg[lowCut[0],lowCut[1],2] = output[lowCut]
		rgbImg[data_i[0],data_i[1],1] = output[data_i]
		rgbImg[hiCut[0],hiCut[1],0] = output[hiCut]
		pylab.clf()
		pylab.imshow(rgbImg, aspect='equal')
		pylab.savefig("rgbtest.png")
		return rgbImg
		
	return output
	
def fitsWriteTest():
	DATAPATH = '/home/madmaze/DATA/LSST/FITS'
	RESPATH  = '/home/madmaze/DATA/LSST/results';
	BASE_N = 141
	#FILENAME = lambda i: '%s/v88827%03d-fz.R22.S11.fits.png' % (DATAPATH,(BASE_N+i))
	FILENAME = lambda i: '%s/v88827%03d-fz.R22.S11.fits' % (DATAPATH,(BASE_N+i))
	xOffset=2000
	yOffset=0
	chunkSize=1000
	yload = lambda i: 1. * readFITS(FILENAME(i))[yOffset:yOffset+chunkSize,xOffset:xOffset+chunkSize].astype(numpy.float32)
	y = yload(0)
	
	# current best seems 450/-50
	#for nonlin in range(0,500,50):
	#	for shift in range(-50,50,10):
	#		asinhScale(y, nonlin, shift, 0, y.max(),"out/test")
	
	nonlin = 450
	shift=-50
	#asinhScale(y, nonlin, shift, 0, y.max(),show=True)
	asinhScale(y, nonlin, shift, 0, 40000, rgb=True)
	
	
def makeHist(inarr,nbins,outfile):
	bins=[]
	bins2=[]
	maxX=max(inarr)
	minX=min(inarr)
	r=abs(minX)+abs(maxX)
	step=r/nbins
	
	s=minX
	while s<maxX:
		bins.append(s)
		s=s+step
	print "sorting into bins.."
	for b in bins:
		cnt=0
		for x in inarr:
			if x>= b and x < b+step:
				cnt=cnt+1
		bins2.append(cnt)
	
	f=open(outfile,"w")
	x=0
	while x < len(bins):
		f.write(str(bins[x])+","+str(bins2[x])+"\n")
		x+=1
	f.close()

if __name__ == "__main__":
	fitsWriteTest()

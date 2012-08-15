#!/usr/bin/env python
import os
import matplotlib.pyplot as plt
import numpy

print "here we go.."
minX = -1383.2088623
maxX = 62395.2773438

rangeX = abs(minX)+abs(maxX)

N=1000

step=rangeX/N

bins=[]
bins2=[]
s=minX
dat = numpy.array([])
while s<maxX:
	bins.append(s)
	dat = numpy.append(dat,s)
	s=s+step
	

def plotit(nonlin,shift, minCut, maxCut):
	fact=numpy.arcsinh((maxCut-minCut)/nonlin)
	print fact
	x=[]
	y=[]
	bins2=[]
	for b in bins:
		xval=b+shift
		if xval > minCut and xval < maxCut:
			x.append(b)
			y.append(numpy.arcsinh(xval/nonlin)/fact)
			
	
	plt.plot(x,y,label=(str(nonlin)+'/'+str(shift)+'/'+str(minCut)+'/'+str(maxCut)))
	#plt.legend(loc=4)
	plt.show()

def asinhScale(data, nonlin, shift, minCut, maxCut):
	output = numpy.array(data, copy=True)
	
	fact=numpy.arcsinh((maxCut-minCut)/nonlin)
	minX=min(data)
	maxX=max(data)
	
	lowCut = numpy.where(output < minCut)
	data_i = numpy.where((output > minCut) & (output < maxCut))
	hiCut = numpy.where(output > maxCut)
	
	output[lowCut] = 0.0
	output[data_i] = numpy.arcsinh(((output[data_i]+shift)/nonlin))/fact
	output[hiCut] = 1.0
	
	#plt.plot(dat.flatten(),output.flatten(),label=(str(nonlin)+'/'+str(shift)+'/'+str(minCut)+'/'+str(maxCut)))
	#plt.legend(loc=4)
	#plt.show()
	return output

plotit(3000.0,100,0,40000)
plotit(300.0,100,0,40000)
plotit(300.0,100,0,60000)

#print dat

asinhScale(dat, 3000.0, 100 ,0, 40000)
asinhScale(dat, 300.0, 100, 0, 40000)
asinhScale(dat, 300.0, 100, 0, 60000)


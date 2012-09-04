#!/usr/bin/env python

import pyfits
import os
import types
import pylab
import numpy
import img_scale
import stopwatch
import imagetools

def readFITS(fitsPath, use_mask=False, norm=False):
    t1 = stopwatch.timer()
    
    t1.start()
    print "loading %s..." % fitsPath
    hdulist = pyfits.open(fitsPath)
    print "load:",t1.elapsed()

    t1.start()
    
    # get image data
    raw_img_data = hdulist[1].data
    
    if use_mask:
        # get and apply mask
        img_mask = hdulist[2].data
        
        print "img_mask:"
        fitsStats(img_mask)
        
        maskIdx = numpy.where(img_mask < 5)
        raw_img_data[maskIdx] = 0.0
    
    hdulist.close()
    
    #width=raw_img_data.shape[0]
    #height=raw_img_data.shape[1]
    
    # cast into float
    raw_img_data = numpy.array(raw_img_data, dtype=numpy.float32)
    
    # shall we cut out the low end?
    lowCut = numpy.where(raw_img_data < 0)
    raw_img_data[lowCut] = 0.0
    
    if norm:
        raw_img_data = raw_img_data/1e5
    
    #maxy = numpy.where(raw_img_data == raw_img_data.max())
    #print raw_img_data.max(), maxy
    #raw_img_data = raw_img_data/raw_img_data.max()
    
    #print raw_img_data.max(), raw_img_data[maxy]
    
    #new_img = img_scale.sqrt(img_data, scale_min=-1.0)
    #new_img = raw_img_data/raw_img_data.max()#asinhScale(raw_img_data, 450, -50, minCut=0.0)
    #print "IM stats (min/max/ave/median):", raw_img_data.min(), raw_img_data.max(), numpy.mean(raw_img_data), numpy.median(raw_img_data)
    #print "-/+:",len(numpy.where(raw_img_data > 0)[0]),len(numpy.where(raw_img_data < 0)[0])
    return raw_img_data.copy()
    
    #return new_img
def asinhScale(data, nonlin, shift, minCut=None, maxCut=None, fname="", rgb=False, fscale=True):
    print "Enter asinhScale.............................."
    minX=data.min()
    maxX=data.max()

    if minCut == None:
        minCut=minX
    if maxCut == None:
        maxCut=maxX
        
    output = numpy.array(data, copy=True)
    
    fact=numpy.arcsinh((maxCut-minCut)/nonlin)
    print "factor:",fact
    
    
    output = output + shift
    
    lowCut = numpy.where(output < minCut)
    data_i = numpy.where((output > minCut) & (output < maxCut))
    hiCut = numpy.where(output > maxCut)
    
    # Zero out low end to avoid negatives
    output[lowCut] = 0.0
    
    # perform asinh and scaling to 0.0-1.0
    if fscale is True:
        output[data_i] = numpy.arcsinh(((output[data_i])/nonlin))/fact
    else:
        output[data_i] = numpy.arcsinh(((output[data_i])/nonlin))
    
    # Cut off high end
    output[hiCut] = 1.0
        
    if rgb is True:
        rgbImg = numpy.zeros((output.shape[0], output.shape[1], 3), dtype=float)
        
        # Make RGB gray scale
        rgbImg[data_i[0],data_i[1],0] = output[data_i]
        rgbImg[data_i[0],data_i[1],1] = output[data_i]
        rgbImg[data_i[0],data_i[1],2] = output[data_i]
        
        # Add in lows in Blue and highs in Red
        # lows are set to 0.0 and therefore wont show up.
        # to make low end visible uncomment:
        #output[lowCut] = 1.0
        rgbImg[lowCut[0],lowCut[1],2] = output[lowCut]
        rgbImg[hiCut[0],hiCut[1],0] = output[hiCut]
        
        if fname != "":
            #pylab.clf()
            #pylab.imshow(rgbImg, aspect='equal')
            #pylab.savefig(fname+"-RGB-"+(str(nonlin)+'-'+str(shift)+'-'+str(minCut)+'-'+str(maxCut))+".png")
            imagetools.imwrite(rgbImg, fname+"_RGB_"+(str(nonlin)+'_'+str(shift)+'_'+str(minCut)+'_'+str(maxCut))+".png")
        print "Leaving asinhScale.........................RGB"
        return numpy.array(rgbImg, copy=True)
    else:
        print "out min/max", output.min(), output.max()
        # Write out image
        if fname != "":
            imagetools.imwrite(output, fname+"_"+(str(nonlin)+'_'+str(shift)+'_'+str(minCut)+'_'+str(maxCut))+".png")
        print "Leaving asinhScale.........................GRAY"
        return numpy.array(output, copy=True)
    
def fitsRGBtest():
    # loads a fits file and then immediately does an asinh scale and saves an RGB .PNG
    DATAPATH = '/home/madmaze/DATA/LSST/FITS'
    RESPATH  = '/home/madmaze/DATA/LSST/results';
    BASE_N = 141
    FILENAME = lambda i: '%s/v88827%03d-fz.R22.S11.fits' % (DATAPATH,(BASE_N+i))
    xOffset=2000
    yOffset=0
    chunkSize=1000
    yload = lambda i: 1. * readFITS(FILENAME(i))[yOffset:yOffset+chunkSize,xOffset:xOffset+chunkSize]
    y = yload(0)
    
    # current best seems 450/-50
    #for nonlin in range(0,1000,50):
        #for shift in range(-2,2):
        #    s=shift/10.0
    #    asinhScale(y, nonlin, -50, minCut=0, maxCut=y.max(),fname="out/testX")
        #imagetools.imwrite(img_scale.asinh(y, scale_min=0.0),"out/test"+str(nonlin)+".png")
    fitsStats(y)
    #asinhScale(y, nonlin, shift, minCut=0, maxCut=40000, fname="test_new", rgb=True)
    nonlin = 450
    shift=-50
    #asinhScale(y, nonlin, shift, 0, y.max(),show=True)
    img = asinhScale(y, nonlin, shift, minCut=0, maxCut=40000, fname="test_orig")
    fitsStats(img)
    #imagetools.imwrite(img, "test_img.png")

def scaleTest():
    # loads a fits file and then immediately does an asinh scale and saves an RGB .PNG
    DATAPATH = '/home/madmaze/DATA/LSST/FITS'
    RESPATH  = '/home/madmaze/DATA/LSST/results';
    BASE_N = 141
    FILENAME = lambda i: '%s/v88827%03d-fz.R22.S11.fits' % (DATAPATH,(BASE_N+i))
    xOffset=2000
    yOffset=0
    chunkSize=1000
    yload = lambda i: 1. * readFITS(FILENAME(i), norm=True)[yOffset:yOffset+chunkSize,xOffset:xOffset+chunkSize]
    y = yload(0)
    print y.max()
    
    fitsStats(y)
    #asinhScale(y, nonlin, shift, minCut=0, maxCut=40000, fname="test_new", rgb=True)
    nonlin = 450
    shift=-50
    y=y*1e5
    #asinhScale(y, nonlin, shift, 0, y.max(),show=True)
    img = asinhScale(y, nonlin, shift, minCut=0,maxCut=40000, fname="test_scaled", rgb=False)
    fitsStats(img)
    #imagetools.imwrite(img, "test_img_Scaled.png")

def maskTest():
    # loads a fits file and then immediately does an asinh scale and saves an RGB .PNG
    DATAPATH = '/home/madmaze/DATA/LSST/FITS'
    RESPATH  = '/home/madmaze/DATA/LSST/results';
    BASE_N = 141
    FILENAME = lambda i: '%s/v88827%03d-fz.R22.S11.fits' % (DATAPATH,(BASE_N+i))
    xOffset=2000
    yOffset=0
    chunkSize=1000
    yload = lambda i: 1. * readFITS(FILENAME(i), use_mask=True, norm=True)[yOffset:yOffset+chunkSize,xOffset:xOffset+chunkSize]
    y = yload(0)
    print y.max()
    
    fitsStats(y)
    #asinhScale(y, nonlin, shift, minCut=0, maxCut=40000, fname="test_new", rgb=True)
    nonlin = 450
    shift=-50
    y=y*1e5
    #asinhScale(y, nonlin, shift, 0, y.max(),show=True)
    img = asinhScale(y, nonlin, shift, minCut=0,maxCut=40000, fname="test_scaled_masked", rgb=False)
    fitsStats(img)
    #imagetools.imwrite(img, "test_img_Scaled.png")
    
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
    
def fitsStats(X):
    # this can be commented out for greater speed..
    print "IM stats (min/max/ave/median):", X.min(), X.max(), numpy.mean(X), numpy.median(X)
    print "-/+:",len(numpy.where(X < 0)[0]),len(numpy.where(X > 0)[0])

if __name__ == "__main__":
    #fitsRGBtest()
    scaleTest()
    maskTest()

#!/usr/bin/env python

from time import time

class timer:
	def __init__(self):
		self.timer=time()
		self.total=0
	
	def reset(self):
		self.timer=time()
		self.total=0
	
	def start(self):
		self.timer=time()
	
	def elapsed(self):
		self.total += (time()-self.timer)
		return (time()-self.timer)
	
	def getTotal(self):
		return self.total

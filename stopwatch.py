#!/usr/bin/env python

from time import time

class timer:
	def __init__(self):
		self.timer=time()
	
	def reset(self):
		self.timer=time()
	
	def elapsed(self):
		return (time()-self.timer)

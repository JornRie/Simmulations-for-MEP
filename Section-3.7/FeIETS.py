#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:45:38 2020

@author: jornrietveld
"""

from time import time

import matplotlib.pyplot as plt
from matplotlib import transforms

import stumpy
from stumpy.datfile import BiasSpec
stumpy.search_path.clear
stumpy.search_path.append(r"/Users/jornrietveld/MepSpyder/201906")

fig,axis = plt.subplots(1,1,figsize = (10,3))

spec = BiasSpec('Spec_032.dat')
lockInY = spec.data['Lock-in_Y (V)']
V = spec.data['Bias calc (V)']
axis.plot(V,lockInY)

axis.set_ylabel(r'Norm. Conductance')
axis.set_xlabel(r'$V_{\mathrm{bias}}$ (V)') 
axis.grid()
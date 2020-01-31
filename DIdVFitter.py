#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:00:56 2019

@author: jornrietveld
"""

from time import time

import matplotlib.pyplot as plt
from matplotlib import transforms

import stumpy
from stumpy.datfile import BiasSpec
stumpy.search_path.clear
stumpy.search_path.append(r"/Users/jornrietveld/MepSpyder/201906")

fig,ax = plt.subplots(1,2,figsize = (10,4))
scaledtrans = transforms.ScaledTranslation(-0.55, -0.15, fig.dpi_scale_trans)
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
for i,axis in enumerate(ax):
    axis.text(0, 1, letters[i], fontsize=14, fontweight="bold", va="bottom", ha="left",
           transform=axis.transAxes + scaledtrans)
    axis.set_ylabel(r'Norm. Conductance')
    axis.set_xlabel(r'$V_{\mathrm{bias}}$ (V)') 
    axis.grid()

start = time()


specList1 = ['Spec_B364.dat','Spec_B365.dat','Spec_B366.dat']
specList2 = ['Spec_D100.dat','Spec_D101.dat','Spec_D102.dat']


for i,spec in enumerate(specList1):
    spec = BiasSpec(spec)
    lockInY = spec.data['Lock-in_Y (V)']
    V = spec.data['Bias calc (V)']
    ax[0].plot(V,lockInY+1.5*i)
    
for i,spec in enumerate(specList2):
    spec = BiasSpec(spec)
    lockInY = spec.data['Lock-in_Y (V)']
    V = spec.data['Bias calc (V)']
    ax[1].plot(V,lockInY+1.5*i)

end = time()
print(end-start)
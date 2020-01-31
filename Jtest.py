#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 09:10:16 2019

@author: jornrietveld
"""

from FunctionsSpinHam import Adatom, Structure, TemperatureVariation, Measurement, getHigestProbabillity
from time import time

import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np

start = time()
T_rangeShort = np.logspace(0.5,-0.5,100)
V_bias = 0.002


Fe1 = Adatom(s=2,D=-2.1,E=0.31,g=2.11)
Fe2 = Adatom(s=2,D=-3.6,E=0.31,g=2.11)
Fe3 = Adatom(s=2,D=-2.1,E=0.311,g=2.11)

JList = np.linspace(0.05,0.1,2)

simDict = {'tipPos':0,'Gs':2.56e-7,'G':1.5e-9,'eta':0.5,'u':0.4}

fig,ax = plt.subplots(1,3,figsize=(12,4.5))
          
scaledtrans = transforms.ScaledTranslation(-0.55, -0.15, fig.dpi_scale_trans)
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
for i,axis in enumerate(ax):
    axis.text(0, 1, letters[i], fontsize=14, fontweight="bold", va="bottom", ha="left",
           transform=axis.transAxes + scaledtrans)

for J in JList:
    struct1 = Structure(atoms=[Fe1,Fe2,Fe2,Fe3,], JDict={(0,1):J,(1,2):J,(2,3):J},Bz = 0.2,)
    struct2 = Structure(atoms=[Fe1,Fe2,Fe2,Fe3,], JDict={(0,1):J,(1,2):J,(2,3):J},Bz = 0.6,)
    lifeTimes1 = TemperatureVariation(struct1,T_rangeShort,**simDict).calcLifeTimesFullRates(V_bias)
    lifeTimes2 = TemperatureVariation(struct2,T_rangeShort,**simDict).calcLifeTimesFullRates(V_bias)
    label1 = 'J = ' + ("%.2f" % J) + ', meV Bz = 0.2 T' 
    label2 = 'J = ' + ("%.2f" % J) + ', meV Bz = 0.6 T' 
    
    v1 = struct1.getEigenStates()
    v2 = struct2.getEigenStates()
    #w = struct.getEnergies()
    print(label1)
    print(getHigestProbabillity(v1[:,0],3,2))
    print(label2)
    print(getHigestProbabillity(v2[:,0],3,2))
    
    avgLifetime1 = 2/(lifeTimes1[:,0]+lifeTimes1[:,1])
    avgLifetime2 = 2/(lifeTimes2[:,0]+lifeTimes2[:,1])
    davgdT1 = np.gradient(avgLifetime1,T_rangeShort)
    davgdT2 = np.gradient(avgLifetime2,T_rangeShort)
    d2avgdT1 = np.gradient(np.gradient(avgLifetime1,T_rangeShort),T_rangeShort)
    d2avgdT2 = np.gradient(np.gradient(avgLifetime2,T_rangeShort),T_rangeShort)
    max1 = np.argmax(d2avgdT1)
    max2 = np.argmax(d2avgdT2)
    
    ax[0].semilogy(1/T_rangeShort,avgLifetime1,'--',label = label1)
    ax[0].semilogy(1/T_rangeShort[max1],avgLifetime1[max1],'rs',markerfacecolor='none')
    ax[1].semilogy(T_rangeShort,davgdT1,'--',label = label1)
    ax[1].semilogy(T_rangeShort[max1],davgdT1[max1],'rs',markerfacecolor='none')
    ax[2].semilogy(T_rangeShort,d2avgdT1,'--',label = label1)
    ax[2].semilogy(T_rangeShort[max1],d2avgdT1[max1],'rs',markerfacecolor='none')
    
    ax[0].semilogy(1/T_rangeShort,avgLifetime2,'--',label = label2)
    ax[0].semilogy(1/T_rangeShort[max2],avgLifetime2[max2],'rs',markerfacecolor='none')
    ax[1].semilogy(T_rangeShort,davgdT2,'--',label = label2)
    ax[1].semilogy(T_rangeShort[max2],davgdT2[max2],'rs',markerfacecolor='none')
    ax[2].semilogy(T_rangeShort,d2avgdT2,'--',label = label2)
    ax[2].semilogy(T_rangeShort[max2],d2avgdT2[max2],'rs',markerfacecolor='none')
    
    #ax[1].semilogy(T_rangeShort,2/(lifeTimes4[:,0]+lifeTimes4[:,1]),'*--',label = label)
    
ax[0].set(xlabel=r'$\frac{1}{T}$ $(\frac{1}{k})$', ylabel=r'Tunnelrates (s$^{-1}$)')
ax[1].set(xlabel=r'T(K)', ylabel=r'$\frac{d t}{d T}$ (s\K)')
ax[2].set(xlabel=r'T(K)', ylabel=r'$\frac{d^2 t}{d T^2}$ (s\K$^2$)')
ax[1].legend()#loc='lower right', bbox_to_anchor=(0,0),framealpha = 0.9)
#ax[1].legend()
ax[0].grid()
ax[1].grid()
ax[2].grid()
#ax[1].set_ylim([1.5e6,1e8])

fig.show()

end = time()
print(end-start)

#(1,2):J,(2,3):J}
#
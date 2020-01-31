#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:52:04 2019

@author: jornrietveld
"""

from FunctionsSpinHam import Adatom, Structure, TemperatureVariation, DIdV
from time import time

import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np


start = time()
T_rangeShort = np.logspace(np.sqrt(2),0,20)
V_bias = 0.002
V_range = np.linspace(0.005,0.020,200)

Fe1 = Adatom(s=2,D=-2.1,E=0.31,g=2.11)
Fe2 = Adatom(s=2,D=-3.6,E=0.31,g=2.11)
Fe3 = Adatom(s=2,D=-2.1,E=0.311,g=2.11)

colorString = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9',]

J_list = [0.7,1.0,1.3]

fig,ax = plt.subplots(1,2,figsize = (9,4))
scaledtrans = transforms.ScaledTranslation(-0.55, -0.15, fig.dpi_scale_trans)
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
for i,axis in enumerate(ax):
    axis.text(0, 1, letters[i], fontsize=14, fontweight="bold", va="bottom", ha="left",
           transform=axis.transAxes + scaledtrans)
    axis.set_ylabel(r'Switching rate ($s^{-1}$)')
    #axis.set_xlabel(r'$\frac{1}{T}$ $(\frac{1}{k})$') 
    axis.grid()

simDict = {'tipPos':0,'Gs':1.452e-7,'G':1.5e-9,'u':0,'eta':0,}

for i,J in enumerate(J_list):
    fourFe = Structure(atoms=[Fe1,Fe2,Fe2,Fe3], JDict={(0,1):J,(1,2):J,(2,3):J},Bz = 3)
    lifeTimes = TemperatureVariation(fourFe,T_rangeShort,**simDict).calcLifeTimesFullRates(V_bias)
    dIdV1 = DIdV(fourFe,V_range,tipPos=0,T=1).getdIdV()
    dIdV2 = DIdV(fourFe,V_range,tipPos=1,T=1).getdIdV()
    
    ax[0].semilogy(1/T_rangeShort,2/(lifeTimes[:,0]+lifeTimes[:,1]),'--s',label = 'J = ' + str(J),markerfacecolor='none')
    ax[1].plot(V_range/1e-3,dIdV1/dIdV1[0],'-'+colorString[i],label = 'J = ' + str(J),markerfacecolor='none')
    ax[1].plot(V_range/1e-3,dIdV2/dIdV2[0] + 0.3,'-'+colorString[i],)

ax[0].set_ylabel(r'Switching rate ($s^{-1}$)')
ax[0].set_xlabel(r'$\frac{1}{T}$ $(\frac{1}{k})$',fontsize=14)

ax[1].set_ylabel(r'$\frac{dI}{dV}$(norm.)')
ax[1].set_xlabel(r'$V$(mV)')
ax[1].set_xlim(5,20)

for axis in ax:
    axis.grid()
    axis.legend()

end = time()
print(end-start)
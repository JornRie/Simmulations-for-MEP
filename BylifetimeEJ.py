#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:16:58 2020

@author: jornrietveld
"""

from FunctionsSpinHam import Adatom, Structure, BigStructure, Measurement, getHigestProbabillity,meV,kb
from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import transforms
import numpy as np

start = time()
T_rangeShort = np.logspace(np.sqrt(2),0,20)
V_bias = 0.001
#V_range = np.linspace(0.005,0.020,200)

T = 1

FielDir = 'By' # 'Bx', 'By' or 'Bz'

Fe1 = Adatom(s=2,D=-2.1,E=0.31,g=2.11)
Fe2 = Adatom(s=2,D=-3.6,E=0.31,g=2.11)
Fe3 = Adatom(s=2,D=-3.7,E=0.31,g=2.11)
Fe4 = Adatom(s=2,D=-2.3,E=0.31,g=2.11)

n = 625

BList = np.linspace(0,11,34)

J = 0.7

JList = np.linspace(0.4,2.2,10)
EList = np.linspace(0.16,0.61,10)

simDict = {'tipPos':0,'Gs':1.452e-8,'G':5e-9,'u':0,'eta':0.0,'b0':0,'T':T}
          
fig,ax = plt.subplots(1,2,figsize=(12,5))
scaledtrans = transforms.ScaledTranslation(-0.55, -0.15, fig.dpi_scale_trans)
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
for i,axis in enumerate(ax):
    axis.text(0, 1, letters[i], fontsize=14, fontweight="bold", va="bottom", ha="left",
           transform=axis.transAxes + scaledtrans)
    
lifetimes = np.zeros((2,len(BList),n)) 

extremeB = np.zeros((len(JList),2))

for i,J4 in enumerate(JList):
    tempDict = {'atoms':[Fe1,Fe2,Fe3,Fe4], 
                'JDict':{(0,1):J4,(1,2):J4,(2,3):J4},
                'Bz': 0.1}
    for j,B in enumerate(BList):
        tempDict[FielDir] = B
        struct = Structure(**tempDict)
        v = struct.getEigenStates()
        print(getHigestProbabillity(v[:,0],3,2))
        lifetimes[0,j,:] = Measurement(struct,**simDict).calcLifeTimesFullRates(V_bias)
    
    avgLifetime = (lifetimes[0,:,0]+lifetimes[0,:,0])/2
    
    extremeB[i,0] = BList[np.argmin((avgLifetime))]
    
    label = 'J= ' + ("%.2f" % J4) + ' meV'
    if (i == 1 or i == 4 or i == 6):
        #plt.figure()
        #plt.title(label)
        ax[0].semilogy(BList,avgLifetime,'--s',label = label,markerfacecolor='none')


axins = inset_axes(ax[0], width=2, height=1.1, 
                   bbox_to_anchor=(.2, .4, .8, .6),
                   bbox_transform=ax[0].transAxes,loc=1, )

axins.plot(JList,extremeB[:,0],'--sk',markerfacecolor='none')  
axins.set_xlabel('$J$ (meV)')
axins.set_ylabel('$B_{\mathrm{min}}$ (T)')  
#axins.grid()

for i,E in enumerate(EList):
    Fe1a = Adatom(s=2,D=-2.1,E=E,g=2.11)
    Fe2a = Adatom(s=2,D=-3.6,E=E,g=2.11)
    Fe3a = Adatom(s=2,D=-3.7,E=E,g=2.11)
    Fe4a = Adatom(s=2,D=-2.3,E=E,g=2.11)
    
    tempDict = {'atoms':[Fe1a,Fe2a,Fe3a,Fe4a], 
                'JDict':{(0,1):J,(1,2):J,(2,3):J},
                'Bz': 0.1}
    
    for j,B in enumerate(BList):
        tempDict[FielDir] = B
        struct = Structure(**tempDict)
        v = struct.getEigenStates()
        print(getHigestProbabillity(v[:,0],3,2))
        lifetimes[1,j,:] = Measurement(struct,**simDict).calcLifeTimesFullRates(V_bias)
    
    avgLifetime = (lifetimes[1,:,0]+lifetimes[1,:,0])/2
    
    extremeB[i,1] = BList[np.argmin(np.abs(avgLifetime))]
    
    label = 'E= ' + ("%.2f" % E) + ' meV'
    #ax[1].semilogy(BList,avgLifetime,'--s',label = label,markerfacecolor='none')
    if (i == 1 or i == 3 or i == 5):
        #plt.figure()
        #plt.title(label)
        ax[1].semilogy(BList,avgLifetime,'--s',label = label,markerfacecolor='none')

axins2 = inset_axes(ax[1], width=2, height=1.1, 
                    bbox_to_anchor=(.2, .4, .8, .6),
                    bbox_transform=ax[1].transAxes,loc=1, )
axins2.plot(EList,extremeB[:,1],'--sk',markerfacecolor='none') 
axins2.set_xlabel('$E$ (meV)')
axins2.set_ylabel('$B_{\mathrm{min}}$ (T)') 
#axins2.grid()

for axis in ax:
    axis.legend()
    #axis.grid()
    axis.set_ylabel('Lifetime (s)') 
    
ax[1].set_xlabel('$B$ (T)') 
ax[0].set_xlabel('$B$ (T)') 

fig.show()
fig.tight_layout()

end = time()
print(end-start)   
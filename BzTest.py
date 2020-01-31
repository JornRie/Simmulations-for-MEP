#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:44:34 2019

@author: jornrietveld
"""

from FunctionsSpinHam import Adatom,Structure,TemperatureVariation,getHigestProbabillity
from time import time

import matplotlib.pyplot as plt
import numpy as np

start = time()
T_rangeShort = np.logspace(0.6,-0.5,100)
V_bias = 0.002

JList = [-0.15,-0.10,-0.05,0,0.05,0.1,0.15]#np.linspace(-0.0,0.15,4)

Fe1 = Adatom(s=2,D=-2.11,E=0.31,g=2.11)
Fe2 = Adatom(s=2,D=-3.6,E=0.31,g=2.11)
Fe3 = Adatom(s=2,D=-3.6,E=0.31,g=2.11)
Fe4 = Adatom(s=2,D=-2.1,E=0.311,g=2.11)

BzList = np.linspace(0.0,1.0,26)

simDict = {'tipPos':0,'Gs':2.56e-7,'G':1.5e-9,'eta':0.5,'u':0.4}

#fig,ax = plt.subplots(1,2,figsize=(10,6))
fig1,ax1 = plt.subplots(1,1)

maxPoint = np.zeros(len(BzList))

for J in JList:
    #fig,ax = plt.subplots(1,2,figsize=(10,6))
    for i, Bz in enumerate(BzList):
        struct = Structure(atoms=[Fe1,Fe2,Fe2,Fe4,], JDict={(0,1):J,(1,2):J,(2,3):J},Bz = Bz,)
        lifeTimes4 = TemperatureVariation(struct,T_rangeShort,**simDict).calcLifeTimesFullRates(V_bias)
        
        v = struct.getEigenStates()
        w = struct.getEnergies()
        label = ("Bz = %.2f" % Bz)
        print(label)
        print(getHigestProbabillity(v[:,0],3,2))
        print(getHigestProbabillity(v[:,1],3,2))
        avgLifetime = 2/(lifeTimes4[:,0]+lifeTimes4[:,1])
        dtdT = np.gradient(np.gradient(avgLifetime,T_rangeShort),T_rangeShort)
        maxPoint[i] = T_rangeShort[np.argmax(dtdT)]
        #ax[0].semilogy(1/T_rangeShort,avgLifetime,'*--',label = label)
        #ax[1].semilogy(T_rangeShort,dtdT,'*--',label = label)
         
    label = ("J = %.2f" % J) + ' meV'
    print(label)
    if (J < 0):
        sss = 's:'  
    elif (J == 0): 
        sss = 'o:' 
    else:
        sss = '^:'
    ax1.plot(BzList,maxPoint,sss,label = label ,markerfacecolor = 'none',lw = 1.2)
    ax1.set(ylabel='Temperture with max 2nd derivative (K)',xlabel='Bz(T)')

ax1.legend(loc='center right')
ax1.grid()

fig1.show()

end = time()
print(end-start)

#ax[0].semilogy(1/T_rangeShort,avgLifetime,'*--',label = label)
#ax[1].semilogy(T_rangeShort,dtdT,'*--',label = label)

#ax[0].set(xlabel=r'$\frac{1}{T}$ $(\frac{1}{k})$', ylabel=r'Switching rate ($s^{-1}$)')
#ax[1].set(xlabel=r'T(K)', ylabel=r'Switching rate ($s^{-1}$)')
#ax[0].legend()
#ax[1].legend()
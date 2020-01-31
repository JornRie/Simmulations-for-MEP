1#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 15:54:12 2019

@author: jornrietveld
"""

from FunctionsSpinHam import Adatom, Structure, BigStructure, TemperatureVariation
from time import time

import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np

start = time()
T_rangeShort = np.logspace(1,0,30)
V_bias = 0.002

fig,ax = plt.subplots(1,2,figsize = (10,4))
scaledtrans = transforms.ScaledTranslation(-0.55, -0.15, fig.dpi_scale_trans)
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
for i,axis in enumerate(ax):
    axis.text(0, 1, letters[i], fontsize=14, fontweight="bold", va="bottom", ha="left",
           transform=axis.transAxes + scaledtrans)

Fe1 = Adatom(s=2,D=-2.1,E=0.31,g=2.11)
Fe2 = Adatom(s=2,D=-3.6,E=0.31,g=2.11)
Fe3 = Adatom(s=2,D=-2.1,E=0.311,g=2.11)

J = 0.7

valueDict = {'k' : np.arange(5,400,10)}

simDict = {'tipPos':0,'Gs':2.56e-6,'G':1.5e-9,'eta':0,'u':0}
structDict = {'atoms':[Fe1,Fe2,Fe2,Fe3], 'JDict':{(0,1):J,(1,2):J,(2,3):J},'Bz':3,}


fourFe1 = Structure(**structDict)
fourFe2 = BigStructure(**structDict,n=5,k = 625)
fourFe3 = BigStructure(**structDict,n=5,k = 10)

lifeTimes1 = TemperatureVariation(fourFe1,T_rangeShort,**simDict).calcLifeTimesFullRates(V_bias)
lifeTimes2 = TemperatureVariation(fourFe2,T_rangeShort,**simDict).calcLifeTimesFullRates(V_bias)
lifeTimes3 = TemperatureVariation(fourFe3,T_rangeShort,**simDict).calcLifeTimesFullRates(V_bias)

avgLifetime1 = 2/(lifeTimes1[:,0]+lifeTimes1[:,1])

for key in valueDict.keys():
    error = np.zeros((3,len(valueDict[key])))
    for j,value in enumerate(valueDict[key]):
        tempdict = structDict
        tempdict[key] = value
        structList = [BigStructure(**structDict,n=2),
                      BigStructure(**structDict,n=10),
                      BigStructure(**structDict,n=30)]
        for i,struct in enumerate(structList):
            lifeTime = TemperatureVariation(struct,T_rangeShort,**simDict).calcLifeTimesFullRates(V_bias)
            avgLifetime = 2/(lifeTime[:,0]+lifeTime[:,1])
            diffPerc = np.abs(avgLifetime1-avgLifetime)/avgLifetime1
            error[i,j] = np.sum(diffPerc > 0.2)
    ax[1].plot(valueDict[key],error[0],'o',markerfacecolor='none',label = 'n =  2')
    ax[1].plot(valueDict[key],error[1],'^',markerfacecolor='none',label = 'n = 10')
    ax[1].plot(valueDict[key],error[2],'s',markerfacecolor='none',label = 'n = 30')
    ax[1].grid()
    ax[1].set(ylabel = r'number of points with error > 20%',xlabel = r'Number of significant contributions taken',)
    ax[1].legend()
        
        

ax[0].semilogy(1/T_rangeShort,avgLifetime1 ,'#4B8BBE',label = 'Normal')
ax[0].semilogy(1/T_rangeShort,2/(lifeTimes2[:,0]+lifeTimes2[:,1]),'^k',label = 'Less States',markerfacecolor='none')
ax[0].semilogy(1/T_rangeShort,2/(lifeTimes3[:,0]+lifeTimes3[:,1]),'sk',label = 'Highest Probability',markerfacecolor='none')
ax[0].grid()
ax[0].set(ylabel = r'Switching rate ($s^{-1}$)',xlabel = r'$\frac{1}{T}$ $(\frac{1}{k})$',)
#ax[0].set_yticks(np.logspace(7,10,7))
ax[0].legend()
fig.show()

end = time()
print(end-start)
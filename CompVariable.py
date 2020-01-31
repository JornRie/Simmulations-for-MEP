#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:15:03 2020

@author: jornrietveld
"""

from FunctionsSpinHam import Adatom, Structure, TemperatureVariation
from time import time

import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np

start = time()
T_rangeShort = np.logspace(np.sqrt(2),0,20)
V_bias = 0.002

fig,ax = plt.subplots(1,2,figsize = (10,4))
scaledtrans = transforms.ScaledTranslation(-0.55, -0.15, fig.dpi_scale_trans)
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
for i,axis in enumerate(ax):
    axis.text(0, 1, letters[i], fontsize=14, fontweight="bold", va="bottom", ha="left",
           transform=axis.transAxes + scaledtrans)
    axis.set_ylabel(r'Switching rate ($s^{-1}$)')
    axis.set_xlabel(r'$\frac{1}{T}$ $(\frac{1}{k})$',fontsize=14) 
    axis.grid()

linestArray = ['k-', '--sC0', '--sC1', '--sC2', '--sC3']

Fe1 = Adatom(s=2,D=-2.1,E=0.31,g=2.11)
Fe2 = Adatom(s=2,D=-3.6,E=0.31,g=2.11)
Fe3 = Adatom(s=2,D=-2.1,E=0.311,g=2.11)

Fe1D = Adatom(s=2,D=-3.1,E=0.31,g=2.11)
Fe2D = Adatom(s=2,D=-4.6,E=0.31,g=2.11)
Fe3D = Adatom(s=2,D=-3.1,E=0.311,g=2.11)

Fe1E = Adatom(s=2,D=-2.1,E=0.61,g=2.11)
Fe2E = Adatom(s=2,D=-3.6,E=0.61,g=2.11)
Fe3E = Adatom(s=2,D=-2.1,E=0.611,g=2.11)

J = 0.7

J2 = 1.2

np.set_printoptions(precision=2)

structList = [{'atoms':[Fe1,Fe2,Fe2,Fe3],'JDict':{(0,1):J,(1,2):J,(2,3):J},'Bz':3}, 
              {'atoms':[Fe1D,Fe2D,Fe2D,Fe3D],'JDict':{(0,1):J,(1,2):J,(2,3):J},'Bz':3},
              {'atoms':[Fe1E,Fe2E,Fe2E,Fe3E],'JDict':{(0,1):J,(1,2):J,(2,3):J},'Bz':3},
              {'atoms':[Fe1,Fe2,Fe2,Fe3],'JDict':{(0,1):J2,(1,2):J2,(2,3):J2},'Bz':3},]

simList = [{'tipPos':0,'Gs':1.452e-7,'G':1.5e-9,'u':0},
           {'tipPos':0,'Gs':1.452e-5,'G':1.5e-9,'u':0},
           {'tipPos':0,'Gs':1.452e-7,'G':1.5e-6,'u':0},
           {'tipPos':0,'Gs':1.452e-7,'G':1.5e-9,'u':1,},
           {'tipPos':0,'Gs':1.452e-7,'G':1.5e-9,'u':0,'b0':0.75},]

for i,structDict in enumerate(structList):
    struct = Structure(**structDict)
    lifeTime = TemperatureVariation(struct,T_rangeShort,**simList[0]).calcLifeTimesFullRates(V_bias)
    ax[0].semilogy(1/T_rangeShort,2/(lifeTime[:,0]+lifeTime[:,1]),linestArray[i],markerfacecolor = 'none')
  
struct = Structure(**structList[0])    

for i,simDict in enumerate(simList):
    lifeTime = TemperatureVariation(struct,T_rangeShort,**simDict).calcLifeTimesFullRates(V_bias)
    ax[1].semilogy(1/T_rangeShort,2/(lifeTime[:,0]+lifeTime[:,1]),linestArray[i],markerfacecolor = 'none')   

ax[0].legend(['Default',r"$D'=D-1$ meV", r"$E'=0.61$ meV",r"$J'=1.2$ meV"])
ax[1].legend(['Default',r"$G_s'=14$ mS", r"$G'=1.5$ mS",r"$u'=1$ ",r"$b_0'=0.75$"])

fig.show()   
    
    
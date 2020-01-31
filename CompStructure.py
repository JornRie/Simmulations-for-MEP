#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:43:15 2019

@author: jornrietveld
"""

from FunctionsSpinHam import Adatom, Structure, TemperatureVariation
from time import time

import matplotlib.pyplot as plt
import numpy as np

start = time()
T_rangeShort = np.logspace(np.sqrt(2),0,20)
V_bias = 0.002

Fe1 = Adatom(s=2,D=-2.1,E=0.31,g=2.11)
Fe2 = Adatom(s=2,D=-3.6,E=0.31,g=2.11)
Fe3 = Adatom(s=2,D=-2.1,E=0.311,g=2.11)

Fe4 = Adatom(s=2,D=-2.1,E=0.41,g=2.11)
Fe5 = Adatom(s=2,D=-3.6,E=0.41,g=2.11)
Fe6 = Adatom(s=2,D=-2.1,E=0.411,g=2.11)

Fe7 = Adatom(s=2,D=-3.1,E=0.31,g=2.11)
Fe8 = Adatom(s=2,D=-4.6,E=0.31,g=2.11)
Fe9 = Adatom(s=2,D=-3.1,E=0.311,g=2.11)

J = 0.7
J1 = 1.2

fourFe1 = Structure(atoms=[Fe1,Fe2,Fe2,Fe3], JDict={(0,1):J,(1,2):J,(2,3):J},Bz = 3)
fourFe2 = Structure(atoms=[Fe4,Fe5,Fe5,Fe6], JDict={(0,1):J,(1,2):J,(2,3):J},Bz = 3)
fourFe3 = Structure(atoms=[Fe7,Fe8,Fe8,Fe9], JDict={(0,1):J,(1,2):J,(2,3):J},Bz = 3)
fourFe4 = Structure(atoms=[Fe1,Fe2,Fe2,Fe3], JDict={(0,1):J1,(1,2):J1,(2,3):J1},Bz = 3)


simDict = {'tipPos':0,'Gs':1.452e-7,'G':1.5e-9,'u':0,'eta':0,}
lifeTimes1 = TemperatureVariation(fourFe1,T_rangeShort,**simDict).calcLifeTimesFullRates(V_bias)
avgLifetime1 = 2/(lifeTimes1[:,0]+lifeTimes1[:,1])
lifeTimes2 = TemperatureVariation(fourFe2,T_rangeShort,**simDict).calcLifeTimesFullRates(V_bias)
avgLifetime2 = 2/(lifeTimes2[:,0]+lifeTimes2[:,1])
lifeTimes3 = TemperatureVariation(fourFe3,T_rangeShort,**simDict).calcLifeTimesFullRates(V_bias)
avgLifetime3 = 2/(lifeTimes3[:,0]+lifeTimes3[:,1])
lifeTimes4 = TemperatureVariation(fourFe4,T_rangeShort,**simDict).calcLifeTimesFullRates(V_bias)
avgLifetime4 = 2/(lifeTimes4[:,0]+lifeTimes4[:,1])

plt.figure(1)
plt.semilogy(1/T_rangeShort,avgLifetime1,'*--',label = r'Normal')
#plt.text(1/T_rangeShort[-1] + 0.01, avgLifetime1[-2] , r'Normal')
plt.semilogy(1/T_rangeShort,avgLifetime2,'*--',label = r'$ \uparrow E $ ')
plt.text(1/T_rangeShort[-1] + 0.02, avgLifetime2[-2] , r'$ \uparrow E $ ')
plt.semilogy(1/T_rangeShort,avgLifetime3,'*--',label = r'$ \uparrow D $ ')
plt.text(1/T_rangeShort[-1] + 0.02, avgLifetime3[-2] , r'$\uparrow D $ ')
plt.semilogy(1/T_rangeShort,avgLifetime4,'*--',label = r'$ \uparrow J $ ')
plt.text(1/T_rangeShort[-1] + 0.02, avgLifetime4[-2] , r'$\uparrow J $ ')
plt.ylabel(r'Switching rate ($s^{-1}$)')
plt.xlabel(r'$\frac{1}{T}$ $(\frac{1}{k})$',fontsize=14)

plt.grid()
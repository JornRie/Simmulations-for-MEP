#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 22:53:27 2020

@author: jornrietveld
"""

from FunctionsSpinHam import Adatom, Structure, BigStructure, TemperatureVariation
from time import time

import matplotlib.pyplot as plt
import numpy as np

start = time()
T_rangeShort = np.logspace(np.sqrt(2),-1,36)
V_bias = 0.002


Fe1 = Adatom(s=2,D=-2.1,E=0.31,g=2.11)
Fe2 = Adatom(s=2,D=-3.6,E=0.31,g=2.11)
Fe3 = Adatom(s=2,D=-2.1,E=0.311,g=2.11)

J = 0.7

fourFe = Structure(atoms=[Fe1,Fe2,Fe2,Fe3], JDict={(0,1):J,(1,2):J,(2,3):J},Bz = 3)

simDict = {'tipPos':0,'Gs':1.452e-9,'G':1.5e-9,'u':0.3,'eta':0.24,'b0':0}

lifeTimesfull = TemperatureVariation(fourFe,T_rangeShort,**simDict).calcLifeTimesFullRates(V_bias)
lifeTimesRss =TemperatureVariation(fourFe,T_rangeShort,**simDict).getLifeTimes()

plt.figure()
plt.semilogy(1/T_rangeShort,2/(lifeTimesfull[:,0]+lifeTimesfull[:,1]),'--C0',label = 'All rates')
plt.semilogy(1/T_rangeShort,2/(lifeTimesRss[:,0]+lifeTimesRss[:,1]),'--k',label = r'Only $r^{s \to s }$')

plt.grid()
plt.ylabel(r'Switching rate ($s^{-1}$)')
plt.xlabel(r'$\frac{1}{T}$ $(\frac{1}{k})$',fontsize=14)
plt.legend()
plt.show()

end = time()
print(end-start)
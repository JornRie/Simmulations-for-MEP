#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:07:57 2019

@author: jornrietveld
"""

from FunctionsSpinHam import Adatom,Structure,DIdV

import matplotlib.pyplot as plt
import numpy as np

Fe1 = Adatom(s=2,D=-2.1,E=0.31,g=2.11)
Fe2 = Adatom(s=2,D=-3.6,E=0.31,g=2.11)
Fe3 = Adatom(s=2,D=-2.1,E=0.311,g=2.11)
Nv = 201
V_range = np.linspace(-0.03,0.03,Nv)
vPoints = [-14.5,-8.5,8.5,14.5]
monomer = Structure(atoms=[Fe1,Fe2,Fe3],JDict = {(0,1):1.15,(1,2):1.15}, Bz = 2)
mV = 0.001
mu = 1e-6

f, ax = plt.subplots(1, 1,sharex=True,figsize=(8,4))

for i in range(len(monomer.atoms)):
    dIdV1 = DIdV(structure=monomer,tipPos=i,T=0.5,V_range=V_range,u=0.5,eta=0,G=2/3e7,Gs=3.1e-6).getdIdV()
    
    mV_range = V_range/mV
    
    mudIdV1 = dIdV1/mu
    
    ax.plot(mV_range,mudIdV1+0.04*i,'#4B8BBE')

ax.set_xlabel('Voltage (mV)')
ax.set_ylabel(r'$\frac{dI}{dV}(\mu S)$')
ax.set_ylim(0.05,0.2)
ax.set_yticks(np.linspace(0.05,0.2,7))
for x in vPoints:
    ax.axvline(x=x, color='k', linestyle='--',lw = 0.8)
ax.grid()
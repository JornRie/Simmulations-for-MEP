#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 21:01:19 2019

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
monomer = Structure(atoms=[Fe1,Fe2,Fe3],JDict = {(0,1):1.15,(1,2):1.15}, Bz = 2)
mV = 0.001
mu = 1e-6

G_range = [0.09e-6, 0.16e-6, 0.22e-6, 0.32e-6, 0.48e-6]

f, ax = plt.subplots(2, 2,sharex=True,figsize=(12,8))

for i,G in enumerate(G_range):
    dIdV0 = DIdV(structure=monomer,tipPos=0,T=0.5,V_range=V_range,u=0.3,eta=0,G=G,Gs=3.1e-6).getdIdV()
    dIdV1 = DIdV(structure=monomer,tipPos=0,T=0.5,V_range=V_range,u=0.5,eta=0,G=G,Gs=3.1e-6).getdIdV()
    dIdV2 = DIdV(structure=monomer,tipPos=0,T=0.5,V_range=V_range,u=0.7,eta=0,G=G,Gs=3.1e-6).getdIdV()
    dIdV3 = DIdV(structure=monomer,tipPos=0,T=0.5,V_range=V_range,u=0.9,eta=0,G=G,Gs=3.1e-6).getdIdV()
    
    mV_range = V_range/mV
    
    mudIdV0 = dIdV0/mu
    mudIdV1 = dIdV1/mu
    mudIdV2 = dIdV2/mu
    mudIdV3 = dIdV3/mu
    
    ax[0,0].plot(mV_range,mudIdV0,'#4B8BBE')
    ax[0,0].text(mV_range[-35], mudIdV0[-1]+0.01, ("%.2f" % (G/mu)) + r' $\mu S$')
    
    ax[0,1].plot(mV_range,mudIdV1,'#4B8BBE')
    ax[0,1].text(mV_range[-35], mudIdV1[-1]+0.01, ("%.2f" % (G/mu)) + r' $\mu S$')
    
    ax[1,0].plot(mV_range,mudIdV2,'#4B8BBE')
    ax[1,0].text(mV_range[-35], mudIdV2[-1]+0.01, ("%.2f" % (G/mu)) + r' $\mu S$')
    
    ax[1,1].plot(mV_range,mudIdV3,'#4B8BBE')
    ax[1,1].text(mV_range[-35], mudIdV3[-1]+0.01, ("%.2f" % (G/mu)) + r' $\mu S$')
 
    
ax[0,0].set_title('u = 0.3')
#ax[0,0].set_xlabel('Voltage (mV)')
ax[0,0].set_ylabel(r'$\frac{dI}{dV}(\mu S)$')
ax[0,0].set_ylim(0.05,0.8)
ax[0,0].set_yticks(np.linspace(0.1,0.8,8))
ax[0,0].grid()

ax[0,1].set_title('u = 0.5')
#ax[0,1].set_xlabel('Voltage (mV)')
ax[0,1].set_ylabel(r'$\frac{dI}{dV}(\mu S)$')
ax[0,1].set_ylim(0.05,0.8)
ax[0,1].set_yticks(np.linspace(0.1,0.8,8))
ax[0,1].grid()

ax[1,0].set_title('u = 0.7')
ax[1,0].set_xlabel('Voltage (mV)')
ax[1,0].set_ylabel(r'$\frac{dI}{dV}(\mu S)$')
ax[1,0].set_ylim(0.05,0.8)
ax[1,0].set_yticks(np.linspace(0.1,0.8,8))
ax[1,0].grid()

ax[1,1].set_title('u = 0.9')
ax[1,1].set_xlabel('Voltage (mV)')
ax[1,1].set_ylabel(r'$\frac{dI}{dV}(\mu S)$')
ax[1,1].set_ylim(0.05,0.8)
ax[1,1].set_yticks(np.linspace(0.1,0.8,8))
ax[1,1].grid()

#f.tight_layout()
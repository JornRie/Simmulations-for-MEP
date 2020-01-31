#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:53:40 2019

@author: jornrietveld
"""

from FunctionsSpinHam import Adatom, Structure, LocalField
from time import time

import matplotlib.pyplot as plt
import numpy as np

start = time()
Bloc_range = np.linspace(0,2,201)
J = 1.15
B = -1
V_bias = None

Fe1 = Adatom(s=2,D=-2.1,E=0.31,g=2.1)
Fe2 = Adatom(s=2,D=-3.6,E=0.31,g=2.1)
Fe3 = Adatom(s=2,D=-2.1,E=0.311,g=2.1)

trimer = Structure([Fe1,Fe2,Fe3],JDict={(0,1):J,(1,2):J},Bz = B)
localField = LocalField(trimer,Bloc_range,tipPos = 0,u = 0,Gs = 3.1e-6,G = 3e-9,eta = 0.2,b0 = 0.5)

Neelprob, T1 = localField.calcNeelProb(V_bias)

fig, ax = plt.subplots(2,1,sharex=True,figsize=(8,6))
ax[0].semilogy(Bloc_range,Neelprob.T)
ax[1].plot(Bloc_range,T1/1e-6)

ax[0].set_ylabel(r'Probability')
ax[1].set_ylabel(r'$T_1(\mu s)$')
ax[1].set_ylim(-0.1,3)
ax[1].set_yticks(np.linspace(0,3,7))
ax[1].set_xlabel(r'$B_{loc}(T)$')
ax[1].set_xlim(0,2)

end = time()
print(end-start)
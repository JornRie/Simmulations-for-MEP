#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:39:23 2019

@author: jornrietveld
"""

from FunctionsSpinHam import Adatom, Structure, BigStructure, Measurement, getHigestProbabillity,meV,kb
from time import time

import matplotlib.pyplot as plt
import numpy as np

start = time()
T_rangeShort = np.logspace(np.sqrt(2),0,20)
V_bias = 0.001
V_range = np.linspace(0.005,0.020,200)

T = 1

magDir = 'By'  # either 'Bx', 'By' or 'Bz'

Fe1 = Adatom(s=2,D=-2.1,E=0.31,g=2.11)
Fe2 = Adatom(s=2,D=-3.6,E=0.31,g=2.11)
Fe3 = Adatom(s=2,D=-3.7,E=0.31,g=2.11)
Fe4 = Adatom(s=2,D=-2.3,E=0.31,g=2.11)

J4 = 0.7 

n = 625

structDict = {'atoms':[Fe1,Fe2,Fe3,Fe4],#Fe2,Fe2,Fe2,Fe4], 
                  'JDict':{(0,1):J4,(1,2):J4,(2,3):J4},
                  'n' : n, 'Bz': 0.1,}

                         #(3,4):0.05,(2,5):0.05,(1,6):0.05,(0,7):0.05,
                         #(4,5):J4,(5,6):J4,(6,7):J4}, 'k' : 600,


valuesDict = {magDir : np.linspace(0,10,21)}#[0,0.2,0.4,0.5,0.6,0.8,1.0,1.2,1.5,1.8,2.1,2.4,2.7,3.0]}#,3.5,4,5,6,7,8,9,10,11,12,13]}


simList = [{'tipPos':0,'Gs':1.452e-8,'G':5e-9,'u':0,'eta':0.0,'b0':0,'T':T},]
           #{'tipPos':0,'Gs':1.452e-8,'G':5e-9,'u':0,'eta':0.1,'b0':0,'T':T},
           #{'tipPos':0,'Gs':1.452e-8,'G':5e-9,'u':0,'eta':0.25,'b0':0,'T':T},
           #{'tipPos':0,'Gs':1.452e-8,'G':5e-9,'u':0,'eta':0.5,'b0':0,'T':T},
           #]

fig, ax = plt.subplots(3,1,figsize = (5,10),)
lifetimes = np.zeros((len(simList),len(valuesDict[magDir]),n)) 
overlap = np.zeros(len(valuesDict[magDir]))
dE = np.zeros((len(valuesDict[magDir]),5),dtype = complex)

for key in valuesDict.keys():
    for j,value in enumerate(valuesDict[key]):
        tempDict = structDict.copy()
        tempDict[key] = value
        struct = Structure(**tempDict)
        v = struct.getEigenStates()
        w = struct.getEnergies()
        overlap[j] = np.sum(np.abs(v[:,0]*v[:,1])**2)
        dE[j,:] = (w[0:5]-w[0])/meV
        print(getHigestProbabillity(v[:,0],3,2))
        for i,simDict in enumerate(simList):
            lifetimes[i,j,:] = Measurement(struct,**simDict).calcLifeTimesFullRates(V_bias)
            
    ax[0].semilogy(valuesDict[key],lifetimes[0,:,0],'--sC0',label = r'$\phi_0 \to \phi_1$',markerfacecolor='none')
    ax[0].semilogy(valuesDict[key],lifetimes[0,:,1],'--^C1',label = r'$\phi_1 \to \phi_0$',markerfacecolor='none')
    #ax[0].set_title(r'$\eta = 0.0$')
    ax[0].set_ylabel('lifetime (s)') 
    
    
    ax[1].semilogy(valuesDict[key],overlap,'--sC3',label = 'overlap $\phi_0$ and $\phi_1$',markerfacecolor='none')
    ax[1].set_ylabel('overlap') 

    ax[2].plot(valuesDict[key],dE[:,0],'--sC4',label = 'dE between $\phi_0$ and $\phi_1$',markerfacecolor='none')
    ax[2].plot(valuesDict[key],dE[:,1],'--sC5',label = 'dE between $\phi_0$ and $\phi_2$',markerfacecolor='none')
    #ax[2].plot(valuesDict[key],dE[:,2],'--sC6',label = 'dE between $\phi_0$ and $\phi_2$',markerfacecolor='none')
    #ax[2].plot(valuesDict[key],dE[:,3],'--sC7',label = 'dE between $\phi_0$ and $\phi_2$',markerfacecolor='none')
    #ax[2].plot(valuesDict[key],dE[:,4],'--sC8',label = 'dE between $\phi_0$ and $\phi_2$',markerfacecolor='none')
    ax[2].set_ylabel(r'$\Delta E$(meV)') 
    
    #ax[0,1].semilogy(valuesDict[key],lifetimes[1,:,0],'--sC0',label = r'$\phi_0 \to \phi_1$',markerfacecolor='none')
    #ax[0,1].semilogy(valuesDict[key],lifetimes[1,:,1],'--^C1',label = r'$\phi_1 \to \phi_0$',markerfacecolor='none')
    #ax[0,1].set_title(r'$\eta = 0.1$')
    #ax[1,0].semilogy(valuesDict[key],lifetimes[2,:,0],'--sC0',label = r'$\phi_0 \to \phi_1$',markerfacecolor='none')
    #ax[1,0].semilogy(valuesDict[key],lifetimes[2,:,1],'--^C1',label = r'$\phi_1 \to \phi_0$',markerfacecolor='none')
    #ax[1,0].set_title(r'$\eta = 0.25$')
    #ax[1,1].semilogy(valuesDict[key],lifetimes[3,:,0],'--sC0',label = r'$\phi_0 \to \phi_1$',markerfacecolor='none')
    #ax[1,1].semilogy(valuesDict[key],lifetimes[3,:,1],'--^C1',label = r'$\phi_1 \to \phi_0$',markerfacecolor='none')            
    #ax[1,1].set_title(r'$\eta = 0.5$')

#for axs in ax:
for axis in ax:
    axis.legend()
    axis.grid()
    axis.set_xlabel('B (T)') 
    #axis.set_ylabel('lifetime (s)') 

    
#fig1, ax1 = plt.subplots(1,1,figsize = (5,4))

#stimate = -overlap*dE[:,0]/(np.exp(-dE[:,0]/(kb*T))-1)

#ax1.plot(valuesDict['Bx'],estimate)

fig.show()
fig.tight_layout()

#fig1.show()

end = time()
print(end-start)              
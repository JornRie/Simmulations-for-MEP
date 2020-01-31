#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:03:20 2019

@author: jornrietveld
"""

from FunctionsSpinHam import Adatom, Structure, BigStructure, TemperatureVariation
from time import time

import matplotlib.pyplot as plt
import numpy as np

start = time()
T_rangeShort = np.logspace(1,0,16)
V_bias = 0.002

#Dividing by meV because I am stupid
#Btip = 0.0
#Jtip = 2.11*Btip*muB/meV

Fe1 = Adatom(s=2,D=-2.1,E=0.31,g=2.11)
Fe2 = Adatom(s=2,D=-3.6,E=0.31,g=2.11)
Fe3 = Adatom(s=2,D=-2.1,E=0.311,g=2.11)
Fe4 = Adatom(s=2,D=-2.1,E=0.31,g=2.11)
Fe5 = Adatom(s=2,D=-3.6,E=0.31,g=2.11)
Fe6 = Adatom(s=2,D=-2.1,E=0.311,g=2.11)

Fe7 = Adatom(s=2,D=-1.5,E=0.31,g=2.11)

Jin = 0.65
Jout = 0.75
J = 0.7
J4 = J#0.43
J6 = J#0.65
J8 = J#1.03

fourFe = Structure(atoms=[Fe1,Fe2,Fe2,Fe3], JDict={(0,1):J,(1,2):J,(2,3):J},Bz = 3)

sixFeB = BigStructure(atoms=[Fe1,Fe2,Fe2,Fe2,Fe2,Fe3], 
                  JDict={(0,1):J6,(1,2):J6,(2,3):J6,(3,4):J6,(4,5):J6}
                  ,Bz = 3,n = 10, k = 30)#,Leads = {0:Lead(Jtip,1)})

eightFe = BigStructure(atoms=[Fe1,Fe2,Fe2,Fe2,Fe2,Fe2,Fe2,Fe3], 
                  JDict={(0,1):J8,(1,2):J8,(2,3):J8,(3,4):J8,(4,5):J8,(5,6):J8,(6,7):J8}
                  ,Bz = 3,n = 10, k = 30)#,Leads = {0:Lead(Jtip,1)})

eightFe2x4 = BigStructure(atoms=[Fe4,Fe5,Fe5,Fe4,Fe4,Fe5,Fe5,Fe6], 
                  JDict={(0,1):J4,(1,2):J4,(2,3):J4,
                         (3,4):0.05,(2,5):0.05,(1,6):0.05,(0,7):0.05,
                         (4,5):J4,(5,6):J4,(6,7):J4}
                  ,Bz = 3,n = 10, k = 30)#,Leads = {0:Lead(Jtip,1)})

simList = [{'tipPos':0,'Gs':1.452e-9,'G':1.5e-9,'u':0.3,'eta':0.24,'b0':0},]
           #{'tipPos':2,'Gs':1.452e-7,'G':1.5e-9,'u':0,'eta':0.24},
           #{'tipPos':3,'Gs':1.452e-7,'G':1.5e-9,'u':0,'eta':0.24},
           #{'tipPos':0,'Gs':1.452e-8,'G':1.5e-9,'u':0,'eta':0.4},
           #{'tipPos':1,'Gs':1.452e-8,'G':1.5e-9,'u':0,'eta':0.4},
           #{'tipPos':2,'Gs':1.452e-8,'G':1.5e-9,'u':0,'eta':0.4},
           #{'tipPos':3,'Gs':1.452e-8,'G':1.5e-9,'u':0,'eta':0.4},
           #{'tipPos':0,'Gs':1.452e-9,'G':1.5e-9,'u':0,'eta':0.4},
           #{'tipPos':1,'Gs':1.452e-9,'G':1.5e-9,'u':0,'eta':0.4},
           #{'tipPos':2,'Gs':1.452e-9,'G':1.5e-9,'u':0,'eta':0.4},
           #{'tipPos':3,'Gs':1.452e-9,'G':1.5e-9,'u':0,'eta':0.4},
           #]

for simDict in simList:

    #lifeTimes4 = TemperatureVariation(fourFe,T_rangeShort,**simDict).calcLifeTimesFullRates(V_bias)
    lifeTimes6B = TemperatureVariation(sixFeB,T_rangeShort,**simDict).calcLifeTimesFullRates(V_bias)
    lifeTimes8 = TemperatureVariation(eightFe,T_rangeShort,**simDict).calcLifeTimesFullRates(V_bias)
    lifeTimes2x4 = TemperatureVariation(eightFe2x4,T_rangeShort,**simDict).calcLifeTimesFullRates(V_bias)
    
    plt.figure(figsize=(6,4))
    #plt.semilogy(1/T_rangeShort,1/lifeTimes8[:,0],'*',label = '1x8 A')
    #plt.semilogy(1/T_rangeShort,1/lifeTimes2x4[:,0],'*',label = '2x4 A')
    #plt.semilogy(1/T_rangeShort,1/lifeTimes4[:,0],'*',label = '1x4 A')
    #plt.semilogy(1/T_rangeShort,1/lifeTimes6B[:,0],'*',label = '1x6 AB')
    plt.semilogy(1/T_rangeShort,2/(lifeTimes6B[:,0]+lifeTimes6B[:,1]),'--C0',label = '1x6')
    plt.semilogy(1/T_rangeShort,2/(lifeTimes8[:,0]+lifeTimes8[:,1]),'--C1',label = '1x8')
    plt.semilogy(1/T_rangeShort,2/(lifeTimes2x4[:,0]+lifeTimes2x4[:,1]),'--C2',label = '2x4 ')
    #plt.semilogy(1/T_rangeShort,2/(lifeTimes4[:,0]+lifeTimes4[:,1]),'--k',label = '1x4 average')
    
    #plt.yticks(np.logspace(-4,6,11))
    plt.grid()
    #plt.title(simDict.__repr__())
    plt.ylabel(r'Switching rate ($s^{-1}$)')
    plt.xlabel(r'$\frac{1}{T}$ $(\frac{1}{k})$',fontsize=14)
    plt.legend()
    plt.show()

    end = time()
    print(end-start)
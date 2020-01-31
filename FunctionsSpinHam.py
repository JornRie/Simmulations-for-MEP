# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
#The road to import extra modules
#sys.path.append('d:/users/jorn/appdata/local/programs/python/python37-32/lib/site-packages')
#sys.path.append('/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages')

import numpy as np
import scipy.sparse as sp
from numpy import linalg as LA
from scipy import linalg as SLA
from scipy.sparse import linalg as SSLA
from scipy.sparse import diags
from bunch import Bunch


hbar = 1

#Struct Dict that consist a key and a revence to a save file for all structs
saveStructDict = {}

# Functions for the operators S_+ and S_-
def Splus(s,ms):
    return (s*(s+1)-ms*(ms+1))**(1/2)

def Smin(s,ms):
    return (s*(s+1)-ms*(ms-1))**(1/2)
        
# Functions for the operators S_+^2 and S_-^2
def Splus2(s,ms):
    return Splus(s,ms)*Splus(s,ms+1) 
    
def Smin2(s,ms):    
    return Smin(s,ms)*Smin(s,ms-1)
    
"""
Returns fermidirac distribution
"""
def fermidirac(x,beta):
    return 1/(np.exp(x*(beta))+1)

"""
Analitical result of the integral of fermidirac(x+A)*(1-fermidirac(x+B)) from -inf to inf
"""
def fermiIntAnalitical(A, B, beta):
    dE = A-B
    dEzero = np.abs(dE) ==0 #<= 1e-40
    dEnotZero = np.abs(dE) > 0# 1e-40 
    res = dE.copy()
    res[dEnotZero] = -dE[dEnotZero]/np.expm1(-dE[dEnotZero]*beta)
    res[dEzero] = 1/beta
    return res
 

def indexToState(n,s):
    """Helper class, converts the given index to the coresponding state.
    Parameters:
        n : int = Index to be converted
        s : int = Spin of the atoms in the structure
    Returns:
        Paritial spin state corresponding to index
    """
    if n==0: return ''
    else:
        return indexToState(n//int(2*s+1),s) + str(n%int(2*s+1))
    

def getStateFromIndex(n,N,s):
    """Converts the given index to the coresponding state.
    Parameters:
        n : int = Index to be converted
        N : int = Number of atoms in the structure
        s : int = Spin of the atoms in the structure
    Returns:
        Complete spin state corresponding to index
    """
    S = indexToState(n,s)
    
    while(len(S)<N):
        S = '0' + S
    
    return S

def getStageredMagnetization(n,N,s):
    """Calculates the staggerd mangetization of a certain state 
    Parameters:
            n : int = Index to be converted
            N : int = Number of atoms in the structure
            s : int = Spin of the atoms in the structure
    Returns:
            Stagered magnetization
    """
    S = getStateFromIndex(n,N,s)
    array = np.array([float(i) for i in list(S)])
    array = 2 - array
    value = 0
    for i,s in enumerate(array):
        value += (-1)**i * s
    return value

  
def getStageredMagnetizationForNatoms(N,s):
    """Calculates the staggerd mangetization of a structure of N atoms with spin s.    
    Parameters:
            n : int = Index to be converted
            N : int = Number of atoms in the structure
            s : int = Spin of the atoms in the structure
    Returns:
            Array with the staggerd magnetization for every possible spin state of the structure
    """ 
    N_max = (s*2+1)**N
    array = np.zeros(N_max)
    for i in range(0,N_max):
        array[i] = getStageredMagnetization(i,N,s)
    return array
    

def getHigestContribution(v,k:int = 3,s=2):
    """Finds the k highest contributing states in a eigen vector and returns their state and value.
    Parameters:
            v : array = Eigen vector of a structure with N atoms
            k : int = Number of states to be found
            s : int = Spin of the atoms in the structure
    Returns:
            String with the k highest contributing states and their contribution
    """
    #Get the number of atoms in the chain
    N = int(np.log(len(v))/np.log(2*s+1))
    n_array = np.abs(v).argsort()[-k:][::-1]
    
    S = []
    for i,n in enumerate( n_array):
        temp = getStateFromIndex(n,N,s) + ' with value: ' + str(v[n])
        S.append(temp)
        
    return S

def getHigestProbabillity(v,k:int = 3,s=2):
    """Finds the k states with the highest probability in a eigen vector.
    Parameters:
            v : array = Eigen vector of a structure with N atoms
            k : int = Number of states to be found
            s : int = Spin of the atoms in the structure
    Returns:
            String with the k highest probability states and their probability
    """
    N = int(np.log(len(v))/np.log(2*s+1))
    n_array = np.abs(v).argsort()[-k:][::-1]
    
    S = []
    for i,n in enumerate( n_array):
        temp = getStateFromIndex(n,N,s) + ' with Probabillity: ' + str(np.abs(v[n])**2)
        S.append(temp)
        
    return S

"""
Dictionary holder classes because JSON can only code dict with str keys. These classes ensure other dicts
are also enocded.
"""
class TupleDict(dict): pass

class IntDict(dict): pass

class MixedDict(dict): pass

class Resetable:
    def resetEnergy(self):
        self.v = None
        self.w = None
        self.Y = None
        self.I = None
        self.lifeTimes = None
    

class Adatom:
    #Physcal constants needed for this class and not used before
    global meV,muB
    meV = 1.602e-22      #from meV to joule
    muB = 5.788e-2*meV   #Bohr magneton in meV/T
    
    def __init__(self, s:float, D:float, E:float, g:float):
        """ Adatom class used to save the parameter specific to the atom on surface. This class will also 
        initalize all the spinoperators that belong to this atom
        Parameters:
            s : int = Spin of the atom
            E : float = Anisotropic parameter in meV
            D : float = Anisotropic parameter in meV
            g : float = landé factor 
        """
        self.s = s
        self.D = D
        self.E = E
        self.g = g
        self.ms = np.arange(s,-s-1,-1)
        self.dim = len(self.ms)
        
        self.H = None
        
        self.Sz = diags([self.ms],[0],format='csr')
        self.Sp = diags([Splus(s,self.ms[1:])],[1],format='csr')
        self.Sp2 = diags([Splus2(s,self.ms[2:])],[2],format='csr')
        self.Sm = diags([Smin(s,self.ms[:-1])],[-1],format='csr')
        self.Sm2 = diags([Smin2(s,self.ms[:-2])],[-2],format='csr')
        self.Sx = (self.Sp+self.Sm)/2
        self.Sy = -1j*(self.Sp - self.Sm)/2
        
    def constructH(self,Bx:float,By:float,Bz:float):
        """ Initializes the Hamiltonian of the adatom
        Parameters:
            Bx : float = In-plane transverse magnetic field (T)
            By : float = Out-of-plane transverse magnetic field (T)
            Bz : float = Magnetic field along easy axis (T)
        """
        H = np.zeros([self.dim,self.dim],dtype = complex)
        Dpart = self.D*meV*self.Sz**2
        Epart = self.E*meV/2*(self.Sp2 + self.Sm2)
        Bpart = -self.g*muB*(self.Sx*Bx + self.Sy*By + self.Sz*Bz)
        H = Dpart+Epart+Bpart
        self.H = H
        
    def findEigenValVect(self):
        """ Calculates the eigen values and vectors if Hamiltonian is constructed
        """
        if self.H is None:
            print("No Hamiltonian constructed")
        else:
            self.v,self.w = SLA.eigh(self.H.toarray())
            
    def constructA(self):
        """ Constructs a matrix that can be used to separate between two Néel states
        """ 
        n = self.dim
        diag = np.zeros(n)
        diag[:int(n/2)] = 1
        if (n/2)%1 != 0:
            diag[int(np.floor(n/2))] = 1/np.sqrt(2)
        return diags(diag,0)
    
    def constructB(self):
        """ Constructs a matrix that can be used to separate between two Néel states
        """ 
        n = self.dim
        diag = np.zeros(n)
        diag[int(n/2):] = 1
        if (n/2)%1 != 0:
            diag[int(np.floor(n/2))] = 1/np.sqrt(2)
        return diags(diag,0)
    
    def __repr__(self):
        s = 's'+ str(self.s) + 'D' + str(self.D) + 'E' + str(self.E) + 'g' + str(self.g)
        return s 
    
    def __str__(self):
        s = 's = '+ str(self.s) + ' ,D = ' + str(self.D) + ' ,E = ' + str(self.E) + ' ,g = ' + str(self.g)
        return s
            

class State():
    
    def __init__(self, atomStates:[int], s:int):
        """ State class used to save the parameter specific to the spin state. 
        Parameters:
            atomStates : [int] = a list of all m_s of the spin state
            s : int = Spin of the atom
        """
        self.atomStates = atomStates
        self.index = int(''.join(str(e) for e in atomStates),int(2*s+1))
        self.vec = np.zeros((2*s+1)**len(atomStates),dtype=complex)
        self.vec[self.index] = 1
        
    def compareAtom(self, i:int,s:int):
        """ State class used to save the parameter specific to the spin state. 
        Parameters:
            i : int = the index of the atom to be compared
            s : int = Spin of the atom should compared
        Returns:
            True if the i'th atom has the same spin as s
        """
        return self.atomStates[len(self.atomStates)-1-i] == 2*s
    
class Lead:
    
    def __init__(self, J:float, s:float, switched = False):
        """ Initializes the Lead class, which used to hold the couling strength of the lead with the atom
        Parameters
            J : float = The couplings strengt of the lead (meV)
            s : float = The size of the magnetic moment of the Lead
        """
        self.J = J
        self.s = s
        self.switched = switched
        
    def switch(self):
        self.switched = not self.switched        
            
class Structure(Resetable):
    
    def __init__(self, atoms:[Adatom],JDict=None,Bx:float = 0,By:float = 0,Bz:float = 0,Leads = None,
                 n:int = 5,w = None, v = None, measurements = {}):
        """ Structure class used to save the parameters specific to the Structure. Initializing this class 
        will also initialize all spin operators needed for the structure. Next to that, it creates the 
        matrices that can be used to callopse state to the Néel states
        Parameters:
            atoms : [Adatom] = a list of the Adatoms were the structure constist of
            JDict : {(int,int):float} = Dictionary used to simulated the bonds between two atoms defined
        by the tuple. The strength is defined by the float (meV).
            Bx : float = In-plane transverse magnetic field (T)
            By : float = Out-of-plane transverse magnetic field (T)
            Bz : float = Magnetic field along easy axis (T)
            Leads : {int:Lead} = A lead and the index of the atom it is applied to
            n : int = the number of eigestates taken into account, only needed if Natoms > 5 atoms
            w : array = eigenvalues of the structure, used when structure is loaded from saved structure
            v : array = eigenstates also used when structure is loaded
            measurements : dict = Can be used if structure is saved with measurements
            
        """
        self.atoms = atoms
        self.JDict = JDict
        #N is the number of atoms
        self.N = len(atoms)
        self.Bx = Bx
        self.By = By
        self.Bz = Bz
        self.Leads = Leads
        #number of eigenstates to be calculated, only needed if N atoms > 5
        self.n = n
        self.dimList = []
        #Construct the Hamiltonian for every atom
        for atom in atoms:
            atom.constructH(Bx,By,Bz)
            self.dimList.append(atom.dim)
            
        self.SList = []
        self.collapseMatList = []
        self.collapseMatListB = []
        #Rewrite the spin operators of the atoms to the structure and creates the matices that can be
        #used to collapse to the Néel states.
        for i in range(0,self.N):
            self.SList.append(self.constructSxyz(i))
            A = self.atoms[i].constructA()
            B = self.atoms[i].constructB()
            self.collapseMatList.append(self.constructOperationU_on_i(A,i))
            self.collapseMatListB.append(self.constructOperationU_on_i(B,i))
        
        self.constructH()
        #Returnable values, implementing not complete
        self.key = self.createKey()
        if self.key in saveStructDict.keys():
            dos = "iets"
            
        self.w = w
        self.v = v
        
        #Part needed for data storage
        self.key = self.createKey()
        self.measurements = measurements
            
    def constructH(self):
        """ Constructs the total Hamiltonian of the structure
        """ 
        Htot = self.atoms[0].H
        HJ = self.constructCouplingHJ()
        HLeads = self.constructHLeads()
        for i in range(1,self.N):
            Htot = sp.kron(Htot,np.eye(self.atoms[i].dim)) + sp.kron(sp.eye(Htot.shape[0]),self.atoms[i].H)
        self.H = Htot + HJ + HLeads
        
    def constructHLeads(self):
        """ Helper method used to construct the Leads part of the Hamiltonian can only be used if there is 
        a magnetic field along the easy axis
        Returns:
            The effect of the leads in the Hamiltonian
        """
        Ntot = np.prod(self.dimList)
        HLeads = sp.csr_matrix((Ntot,Ntot))
        if self.Bz != 0:
            sign = self.Bz/np.abs(self.Bz)
        else:
            #Needs implementing
            return HLeads
        if self.Leads is not None:
            for i, Lead in self.Leads.items():
                Sx_i,Sy_i,Sz_i = self.SList[i]
                sLead = Lead.s
                msLead = sign*sLead*(-1)**Lead.switched
                Sx_lead = Splus(sLead,msLead) + Smin(sLead,msLead)
                Sy_lead = -1j*(Splus(sLead,msLead) - Smin(sLead,msLead))
                HLeads += Lead.J*meV*(Sz_i*msLead + Sx_i*Sx_lead + Sy_i*Sy_lead)
        return HLeads
        
    def constructCouplingHJ(self):
        """ Helper method used to construct the Couplings part of the Hamiltonian
        Returns:
            The effect of the coupling to the Hamiltonian
        """
        Ntot = np.prod(self.dimList)
        HJ = sp.csr_matrix((Ntot,Ntot)) 
        
        if self.JDict is None:
            return HJ
        
        for (i,j), Jij in self.JDict.items():
            Sx_i,Sy_i,Sz_i = self.SList[i]
            Sx_j,Sy_j,Sz_j = self.SList[j]
            HJ += Jij*meV*(Sz_i*Sz_j + Sx_i*Sx_j + Sy_i*Sy_j)
        
        return HJ
        
    def constructSxyz(self,i):
        """ Helper method used to rewrite the spin operators of the atoms to the structure basis.
        Parameters:
            i : int = the position of the atom that will be rewritend
        Returns:
            Sx, Sy and Sz of i'th atom written in the structure basis
        """
        dimList = self.dimList
        Nbefore = np.prod(dimList[:i])
        try:
            Nafter = np.prod(dimList[i+1:])
        except:
            Nafter = 1
        Sx_i = sp.kron(sp.kron(sp.eye(Nbefore),self.atoms[i].Sx),sp.eye(Nafter))
        Sy_i = sp.kron(sp.kron(sp.eye(Nbefore),self.atoms[i].Sy),sp.eye(Nafter))
        Sz_i = sp.kron(sp.kron(sp.eye(Nbefore),self.atoms[i].Sz),sp.eye(Nafter))
        
        return [Sx_i,Sy_i,Sz_i]
    
    def constructOperationU_on_i(self,U,i):
        """ Method that can be used created a matrix that can let U work on the i'th atom
        Parameters:
            U : array = Operator which should work on the i'th atom must have dimensions matching to the atom
            i : int = the position of the atom that will be rewritend
        Returns:
            U projected to the structure basis
        """
        dimList = self.dimList
        Nbefore = np.prod(dimList[:i])
        try:
            Nafter = np.prod(dimList[i+1:])
        except:
            Nafter = 1
        A = sp.kron(sp.kron(sp.eye(Nbefore),U),sp.eye(Nafter))        
        return A
    
    def findAllStates(self):
        """ Helper method that calculates the eigenstates and measurements of the structure. Uses n if N > 5
        """
        if self.N > 5:
            self.w,self.v = SSLA.eigsh(self.H,k = self.n, which = 'SA')
            index = self.w.argsort()
            self.w = self.w[index]
            self.v = self.v[:,index]
            return
        else:
            self.w,self.v = SLA.eigh(self.H.toarray())
            
    def switchLead(self,i):
        """ Method that switches the Lead on the i'th atom.
        Parameter:
            i : int = the index of the atom whose lead has to be switched
        """
        try:
            self.Leads[i].switch()
        except:
            return
        self.constructH()
        self.resetEnergy()
        
    def calcY(self, u : float, tipPos : int):
        """ Method that calculates the Y matrices used for the rate equations
        Parameter:
            u : float = the size of the parameter u which stand for elastic tunneling
            tippos : int = the index of the atom where tip is above
        Returns:
            a package with two Y used for the tip interaction and the subtrate interaction
        """
        v = self.getEigenStates()
        Ntot = v.shape[1]
        Ytot = np.zeros((2,2,Ntot,Ntot),dtype = complex)
        Ytip = np.zeros((2,2,Ntot,Ntot),dtype = complex)
        for i in range(0,self.N):
            Sx,Sy,Sz = self.SList[i]
            Sp = (Sx+1j*Sy)/2
            Sm = (Sx-1j*Sy)/2
            y0 = -Sz/2
            y1 = Sm
            y2 = Sp
            y3 = Sz/2
            y = np.array([[y0.toarray(),y1.toarray()],[y2.toarray(),y3.toarray()]])
            first = np.einsum("ijkl,lm->ijkm",y,v)
            thisY = np.einsum("kl,ijlm->ijkm",v.T,first)
            Ytot += thisY
            if i == tipPos:
                Ytip = thisY
                
        U = u*np.eye(Ntot)
        Ytot[0,0,:,:] += U
        Ytot[1,1,:,:] += U
        Ytip[0,0,:,:] += U
        Ytip[1,1,:,:] += U
        
        return Bunch(Ysurf=np.abs(Ytot)**2,Ytip=np.abs(Ytip)**2)
    
    def indexNeelstate(self):
        """ Method that returs the index of the eigenstate that contain the Néel states
        Returns:
            The two indices where the Néel states are.
        """
        s2 = int(2*self.atoms[0].s)
        g0 = ""
        g1 = ""
        for i in range(0,self.N):
            if i%2==0:
                g0 += "0"
                g1 += str(s2)
            else:
                g1 += "0"
                g0 += str(s2)
        return int(g0,int(s2+1)),int(g1,int(s2+1))

    def vectorNeelstate(self):
        """ Method that returs the a vector that indicates the position of the  the Néel states in the 
        eigen state.
        Returns:
            Two zero vectors with a one at the index of an opposit Néel states each.
        """
        s = self.atoms[0].s
        N = self.N
        a,b = self.indexNeelstate()
        g0 = np.zeros(int((2*s+1)**N))
        g1 = np.zeros(int((2*s+1)**N))
        g0[a] = g1[b] = 1
        return g0,g1

        
    def getEigenStates(self):
        """ Getter that returs the eigen states of the Structure.
        Returns:
            The eigen states of the Structure.
        """
        if self.v is not None:
            return self.v
        else:
            self.findAllStates()
            return self.v
        
    def getEnergies(self):
        """ Getter that returs the eigen Energies of the Structure. 
        Returns:
            The eigen energyies of the Structure.
        """
        if self.w is not None:
            return self.w
        else:
            self.findAllStates()
            return self.w
        
    def getStateEnergies(self):
        """ Getter that returs the spin state Energies relevant for the Structure.
        Returns:
            The spin states energies of the Structure.
        """
        v = self.getEigenStates()
        w = self.getEnergies()
        E = abs(v)**2@w
        return E - np.min(E)
    
    def createKey(self):
        """ Helper method that creates a key that defines unique Structures.
        Returns:
            The created key.
        """
        s = ''
        for atom in self.atoms:
            s += repr(atom)   
        s += 'Bx' + str(self.Bx) + 'By' + str(self.By) +'Bz' + str(self.Bz)
        return s
    
    def measure(self,measurementName,**kwargs):
        """ Method that can be used to do a measurements on a Structure
        Parameter:
            measurementName : str = the name corresponding to the class name of the measurement
            kwargs = variables needed to do the measurement
        Returns:
            Information gained from the measurement
        """
        givenClass = getattr(sys.modules[__name__], measurementName)
        measurement = givenClass(self,**kwargs)
        key = measurement.key
        if measurementName in self.measurements.keys():
           
            if key in self.measurements[measurementName]:
                information = measurement.getInformation(self.measurements[measurementName][key])
                self.measurements[measurementName][key].update(information)
            else:
                information = measurement.getInformation(MixedDict({}))
                self.measurements[measurementName][key] = MixedDict(information)
        else:
            information = measurement.getInformation(MixedDict({}))
            self.measurements[measurementName] = TupleDict({key: MixedDict(information)})
        return information
    
    def remeasure(self,measurementName,**kwargs):
        """ Method that can be used to do remeasure a measurement done before on a Structure
        Parameter:
            measurementName : str = the name corresponding to the class name of the measurement
            kwargs = variables needed to do the measurement
        Returns:
            Information gained from the measurement
        """
        givenClass = getattr(sys.modules[__name__], measurementName)
        measurement = givenClass(self,**kwargs)
        key = measurement.key
        if measurementName in self.measurements.keys():
           
            if key in self.measurements[measurementName]:
                information = measurement.getInformation(MixedDict({}))
                self.measurements[measurementName][key].update(information)
            else:
                information = measurement.getInformation(MixedDict({}))
                self.measurements[measurementName][key] = MixedDict(information)
        else:
            information = measurement.getInformation(MixedDict({}))
            self.measurements[measurementName] = TupleDict({key: MixedDict(information)})
        return information
    
    def __repr__(self):
        s = 'structure \n'
        
        for atom in self.atoms:
            s += str(atom) + ' \n'
            
        s += 'Bx = ' + str(self.Bx) + ' ,By = ' + str(self.By) +' ,Bz = ' + str(self.Bz) + ' \n \n'
        s += 'Measurements saved: \n '
        
        measurementDict = self.measurements
        for measurementKind in measurementDict.keys():
            s += measurementKind + ': \n' 
            for meas in  measurementDict[measurementKind].keys():
                unpackedTuple = ('   T = ' + str(meas[0]) + ' ,tipPos = ' + str(meas[1]) + ' ,u = ' + 
                    str(meas[2]) + ' ,eta = ' + str(meas[3]) + ' ,G = ' + str(meas[4]) + ' ,b0 =' +
                    str(meas[5]) + ' ,Gs = ' + str(meas[6]) + ' : \n' )
                s += unpackedTuple
                data = measurementDict[measurementKind][meas]
                for dataPoint in data.keys():
                    if not isinstance(dataPoint,str):
                        s += '      ' + str(dataPoint) + ' = ' + str(data[dataPoint]) + ' \n'
                    
        return s
      

        
class BigStructure(Structure):
    
    def __init__(self, atoms:[Adatom],JDict=None,Bx:float = 0,By:float = 0,Bz:float = 0,Leads = None,
                 n:int = 5,k:int = 8, w = None, v = None, measurements = {}):
        """ Structure class used for bigger structures. Has an inplementation where only the highest k 
        contributions of the ground state and first exited state are taken into account.
        Parameters:
            atoms : [Adatom] = a list of the Adatoms were the structure constist of
            JDict : {(int,int):float} = Dictionary used to simulated the bonds between two atoms defined
        by the tuple. The strength is defined by the float (meV).
            Bx : float = In-plane transverse magnetic field (T)
            By : float = Out-of-plane transverse magnetic field (T)
            Bz : float = Magnetic field along easy axis (T)
            Leads : {int:Lead} = A lead and the index of the atom it is applied to
            n : int = the number of eigestates taken into account
            k : int = the k highest contribution to the eigenstates taken into account
            w : array = eigenvalues of the structure, used when structure is loaded from saved structure
            v : array = eigenstates also used when structure is loaded
            measurements : dict = Can be used if structure is saved with measurements   
        """
        super().__init__(atoms = atoms, JDict = JDict, Bx = Bx, By = By, Bz = Bz,Leads = Leads,n = n,
                 w = w, v = v, measurements = measurements)
        
        self.k = int(k)
        if w or v is None:
            self.findAllStates()
        
        self.getRelevantIndices()
        self.V = self.v[self.relevantIndices,:]
        
    #Finds the indices that are relevant (highest abs contribution) in the first two eigenstates
    def getRelevantIndices(self):
        """ Helper method used to find the highest contributions to the ground state and the first exited
        state.
        """
        n_arrayA = np.abs(self.v[:,0]).argsort()[-self.k:][::-1]
        n_arrayB = np.abs(self.v[:,1]).argsort()[-self.k:][::-1]
        
        print(getStateFromIndex(n_arrayA[0],self.N,2))
        print(getStateFromIndex(n_arrayB[0],self.N,2))
        
        relevantIndices = list(set(n_arrayA.tolist() + n_arrayB.tolist()))

        self.relevantIndices = relevantIndices
    
    def calcY(self, u : float, tipPos : int):
        """ Method that calculates the Y matrices used for the rate equations, specialist to only take the 
        k highest contributions to ground and first exited state into account.
        Parameter:
            u : float = the size of the parameter u which stand for elastic tunneling
            tippos : int = the index of the atom where tip is above
        Returns:
            a package with two Y used for the tip interaction and the subtrate interaction
        """
        v = self.v[self.relevantIndices,:]
        print(v.shape)
        relevantx, relavanty = np.meshgrid(self.relevantIndices,self.relevantIndices)
        print(relevantx.shape)
        Ntot = v.shape[1]
        Ytot = np.zeros((2,2,Ntot,Ntot),dtype = complex)
        Ytip = np.zeros((2,2,Ntot,Ntot),dtype = complex)
        for i in range(0,self.N):
            Sx,Sy,Sz = self.SList[i]
            Sx = Sx.toarray()[relevantx, relavanty]
            Sy = Sy.toarray()[relevantx, relavanty]
            Sz = Sz.toarray()[relevantx, relavanty]
            Sp = (Sx+1j*Sy)/2
            Sm = (Sx-1j*Sy)/2
            y0 = -Sz/2
            y1 = Sm 
            y2 = Sp
            y3 = Sz/2
            y = np.array([[y0,y1],[y2,y3]])
            first = np.einsum("ijkl,lm->ijkm",y,v)
            thisY = np.einsum("kl,ijlm->ijkm",v.T,first)
            Ytot += thisY
            if i == tipPos:
                Ytip = thisY
                
        U = u*np.eye(Ntot)
        Ytot[0,0,:,:] += U
        Ytot[1,1,:,:] += U
        Ytip[0,0,:,:] += U
        Ytip[1,1,:,:] += U
        
        return Bunch(Ysurf=np.abs(Ytot)**2,Ytip=np.abs(Ytip)**2)
    
    def indexNeelstate(self):
        """ Finds the Néel state index in the new relavent contributions basis
        """
        indexA,indexB = super().indexNeelstate()
        
        #self.relevantIndices.append(indexA)
        #self.relavantIndices = list(set(self.relevantIndices))
        indexA = self.relevantIndices.index(indexA)
        
        #self.relevantIndices.append(indexB)
        #self.relavantIndices = list(set(self.relevantIndices))
        indexB = self.relevantIndices.index(indexB)
        
        return indexA,indexB
    
    def findAllStates(self):
        """ Helper method to find the eigen energies and state of the Structure.
        """
        self.w,self.v = SSLA.eigsh(self.H,k = self.n, which = 'SA')
        index = self.w.argsort()
        self.w = self.w[index]
        self.v = self.v[:,index]
    
    def getEigenStates(self):
        """ Getter method that gives the eigen states of the struture for the relevant indeces.
        """
        if self.v is not None:
            return self.v[self.relevantIndices]
        else:
            self.findAllStates()
            return self.v[self.relevantIndices]

    def getStateEnergies(self):
        """ Getter method that gives the n lowest eigen Energies of the struture.
        """
        print('ok')
        v = self.getEigenStates()
        w = self.getEnergies()
        E = abs(v)**2@w
        return E
        
class Measurement(Resetable):
    
    global e_charge, kb
    e_charge = 1.602e-19 #charge of an electron
    kb = 0.08617*meV     #boltzmann constant
    
    def __init__(self, structure:Structure,T:float = 0.5,tipPos:int = 0,u:float=1,eta:float=0,
                 G:float=0.03e-6,b0:float=0,Gs:float=3e-6):
        """ Initializes the Measurement class that can be used to calculate properties of the system that 
        are related to measurements, such as the tunnel rates.
        Parameters:
            structure : Structure = the Structure that is measured
            T : float = temperature at which the measurement is done (K)
            tipPos : int = atom position of the tip
            u : float = model parameter that stands for elastic tunneling
            eta : float = polarization in the tip along the easy axis
            G : float = tunnel conductance between the tip and the substrate
            b0 : float = part of the electrons that does not interact with the atom (S)
            Gs : float = Surface conductance a measure of the interaction between the atom and the substrate (S)
        """
        self.structure = structure
        self.T = T
        self.beta = 1/(kb*T)
        #The atom position of the tip
        self.tipPos:int = tipPos
        #Different parameters related to the current
        self.u = u
        self.eta = eta
        self.G = G
        self.b0 = b0
        self.Gs = Gs
        self.Gac = (1-self.b0)*self.G
        #Returnable values 
        self.Y = None
        self.I = None
        self.lifeTimes = None
        #Save self in structure 
        self.key = (self.T,self.tipPos,self.u,self.eta,self.G,self.b0,self.Gs)
    
    def calcY(self):
        """ Gets the Y matrix from the structure, needed for the rate equations
        """
        self.Y = self.structure.calcY(self.u,self.tipPos)
        
    def calcPmat(self):
        """ Calculates the P matrix that contain the probability of transition between states
        Returns:
            The P matrices for the three different transition
        """
        #Load the to different Y matrices 
        Y = self.getY()
        Ysurf = Y.Ysurf
        Ytip = Y.Ytip
        #Calc the normalisation factor for P matrices
        P0_surf = (Ysurf[0,0,0,0]*(.5-.5*self.eta) + Ysurf[1,1,0,0]*(.5+.5*self.eta))#self.structure.N
        P0_tip = (Ytip[0,0,0,0]*(.5-.5*self.eta) + Ytip[1,1,0,0]*(.5+.5*self.eta))
        #Construct the matrix handeling the diffrent polarizations
        sigmat = np.array([[-.5,-.5],[.5,.5]])
        eta_fac = 0.5 + self.eta*sigmat
        #Calculation of the P matrices
        P_ts = np.einsum("ijkl,ij->kl",Ytip,eta_fac)/P0_tip
        P_st = np.einsum("ijkl,ij->kl",Ytip,eta_fac.T)/P0_tip
        P_ss = np.einsum("ijkl->kl",Ysurf)/P0_surf
        return P_ts,P_st,P_ss
    
    def calcRates(self,v_bias : float = 0):
        """ Calculates the transition rates of structure induced by electrons traveling between the tip 
        and the substrate. 
        Returns:
            Calculated Rates induced by electrons
        """
        #Loading energies and P matrices
        P_ts,P_st,P_ss = self.calcPmat()
        #Take the energies and calculate all the differences
        w = self.structure.getEnergies()
        wx,wy = np.meshgrid(w,w)
        wv = wy - wx
        #Calculation of different rates by given bias
        r_ts = fermiIntAnalitical(+v_bias*e_charge,wv,self.beta)*P_ts*self.Gac/e_charge**2
        r_st = fermiIntAnalitical(-v_bias*e_charge,wv,self.beta)*P_st*self.Gac/e_charge**2
        r_ss = fermiIntAnalitical(0,wv,self.beta)*P_ss*self.Gs/e_charge**2
        return r_ts,r_st,r_ss
    
    def calcI(self,v_bias : float = 0):
        """ Calculates the tunnel current between the tip and the sample.
        Parameters:
            v_bias : float = the voltage applied on the tip
        Returns:
            The current for the set voltage
        """
        r_ts,r_st,r_ss = self.calcRates(v_bias)
        #r_mat is needed to calculate the occupation of the different levels
        r_mat = r_ts + r_st + r_ss
        
        dim = r_mat.shape[0]
        #A and B are needed for steady state calculation
        A = np.ones((dim+1,dim))
        B = np.zeros(dim+1)
        #The last row of B ensures total probability of 1
        B[-1] = 1
        rdif = r_mat - sp.diags(np.sum(r_mat,axis=0),0)
        A[:-1,:] = rdif
        n_mat,_,_,_ = SLA.lstsq(A,B)
        #Calculation of I 
        self.I = e_charge*np.sum(np.sum(r_ts-r_st,axis=0)*n_mat,axis=0) + self.b0*self.G*v_bias
        #Extra Calculation of the life times of the states
        rates = r_ss.copy()
        np.fill_diagonal(rates, 0)
        self.lifeTimes = ((np.sum(rates,axis=1))**-1)
        
    def calcLifeTimes(self):
        """ Calculates the lifetimes of the eigenstate, whitout taking the tip rates in consideration
        """
        #Calculation of different rates to the surfes for each state
        _,_,r_ss = self.calcRates()
        #Calculation of the life times of the states
        rates = r_ss.copy()
        np.fill_diagonal(rates, 0)
        self.lifeTimes = ((np.sum(rates,axis=0))**-1)
        
    def calcLifeTimesFullRates(self,V_bias=0):
        """ Calculates the lifetimes of state when taking the tip and a voltage in consideration.
        Parameters:
            v_bias : float = the voltage applied on the tip
        Returns:
            The lifetimes of states for the set voltage
        """
        #Loading energies and rates matrices
        r_ts,r_st,r_ss = self.calcRates(V_bias)
        #Calculation of the life times of the states
        rates = r_ss + r_ts + r_st
        np.fill_diagonal(rates, 0)
        lifeTimes = ((np.sum(rates,axis=0))**-1)
        
        return lifeTimes

    #Getters to prevent calculation what was calculated before
    def getY(self):
        """ Getter method to get Y
        """
        if self.Y is not None: 
            return self.Y
        else:
            self.calcY()
            return self.Y
        
    def getI(self,v_bias):
        """ Getter method to get I for a certain voltage
        """
        if self.I is not None:
            return self.I
        else:
            self.calcI(v_bias)
            return self.I  
        
    def getLifeTimes(self):
        """ Getter method to get lifetimes without the tip take into account
        """
        if self.lifeTimes is not None:
            return self.lifeTimes
        else:
            self.calcLifeTimes()
            return self.lifeTimes
        
    def getInformation(self,currentInformation):
        """ Method that is called when measure is used in the Structure class
        """
        if 'lifeTime' not in currentInformation.keys():
            currentInformation['lifeTime'] = self.getLifeTimes()
        return currentInformation 
        
class DIdV(Measurement):
    
    def __init__(self, structure:Structure,V_range,T:float = 0.5,tipPos:int = 0,u:float=1,eta:float=0,
                 G:float=0.03e-6,b0:float=0,Gs:float=1e-6):
        super().__init__(structure, T, tipPos, u, eta, G, b0,Gs)
        """ Initializes the Measurement class that can be used to calculate a dIdV
        Parameters:
            structure : Structure = the Structure that is measured
            V_range : array = array with voltages for, which the dIdV should be calculated
            T : float = temperature at which the measurement is done (K)
            tipPos : int = atom position of the tip
            u : float = model parameter that stands for elastic tunneling
            eta : float = polarization in the tip along the easy axis
            G : float = tunnel conductance between the tip and the substrate
            b0 : float = part of the electrons that does not interact with the atom (S)
            Gs : float = Surface conductance a measure of the interaction between the atom and the substrate (S)
        """
        self.V_range = V_range
        self.dIdV = None
    
    def calcRates(self):
        """ Calculates the transition rates of structure induced by electrons traveling between the tip 
        and the substrate, but than for all voltages in V_range. 
        Returns:
            Calculated Rates induced by electrons
        """
        #Loading energies and P matrices
        P_ts,P_st,P_ss = self.calcPmat()
        #Take the energies and calculate all the differences
        w = self.structure.getEnergies()
        wx,wy = np.meshgrid(w,w)
        wv = wy - wx
        #Getting V_range out of self because it is called often
        V_range = self.V_range
        #The matrices for V and W has to change size, so they can be added to each other
        mold = np.ones(wv.shape)
        wv = np.einsum("ij,k->ijk",wv,np.ones(V_range.shape))
        Vv = np.einsum("ij,k->ijk",mold,V_range)
        #Calculated the fermi part of the intergrals for r_ts,r_st and r_ss
        r_tsFermi = fermiIntAnalitical(+Vv*e_charge,wv,self.beta)
        r_stFermi = fermiIntAnalitical(-Vv*e_charge,wv,self.beta)
        r_ssFermi = fermiIntAnalitical(0,wv,self.beta)
        #Multipli this term with the diffrent probabilities
        r_ts = np.einsum("ijk,ij->ijk",r_tsFermi,P_ts)*self.Gac/e_charge**2
        r_st = np.einsum("ijk,ij->ijk",r_stFermi,P_st)*self.Gac/e_charge**2
        r_ss = np.einsum("ijk,ij->ijk",r_ssFermi,P_ss)*self.Gs/e_charge**2
        return r_ts,r_st,r_ss
        
    def calcI(self):
        """ Calculates the tunnel current based on the rates for all the voltages in V_range.
        """
        #Getting V_range out of self because it is called often
        V_range = self.V_range
        r_ts,r_st,r_ss = self.calcRates()
        #sum the different rates to one matrix
        r_mat = r_ts + r_st + r_ss
        #making space to save the diffrent n_i values
        dim = r_mat.shape[0]
        n_mat = np.zeros((dim,len(V_range)))
        #Matrices for finding n
        A = np.ones((dim+1,dim))
        B = np.zeros(dim+1)
        B[-1] = 1
        #calculate all the diffrent n_i at t=inf for the different voltages
        for k in range(0,len(V_range)):
            rdif = r_mat[:,:,k] - sp.diags(np.sum(r_mat[:,:,k],axis=0),0)
            A[:-1,:] = rdif
            n_mat[:,k],_,_,_ = SLA.lstsq(A,B)
            
        self.n_mat = n_mat
        #Calculate I and dIdV  
        self.I = e_charge*np.sum(np.sum(r_ts-r_st,axis=0)*n_mat,axis=0) + self.b0*self.G*V_range
        #Extra Calculation of the life times of the states
        rates = r_ss[:,:,int(len(V_range)/2)]#np.mean(r_ss[:,:,:],axis=2)        
        np.fill_diagonal(rates, 0)
        self.lifeTimes = ((np.sum(rates,axis=0))**-1)

        
    def calcdIdV(self):
        """ Calculates the dIdV for all the voltages in V_bias.
        """
        I = self.getI()
        self.dIdV = np.gradient(I,self.V_range)
        
    def getI(self):
        """ Getter method for the current.
        """
        if self.I is not None:
            return self.I
        else:
            self.calcI()
            return self.I  
        
    def getdIdV(self):
        """ Getter method for the dIdV.
        """
        if self.dIdV is not None:
            return self.dIdV
        else:
            self.calcdIdV()
            return self.dIdV
        
    def getInformation(self,currentInformation):
        """ Method called by measure from within Structure, gets the information for the given voltages.
        """
        keys = currentInformation.keys()
        updateInformation = {}
        newV_range = []
        for v_bias in self.V_range:
            if v_bias not in keys:
                newV_range.append(v_bias)
            else:
                updateInformation[v_bias] = currentInformation[v_bias] 
        
        if len(newV_range) != 0 : 
            print('Done')
            self.V_range = np.array(newV_range)
            dIdV = self.getdIdV()
            I = self.getI()
            for i,v_bias in enumerate(newV_range):
                updateInformation[v_bias] = (dIdV[i],I[i])
        
        updateInformation = super().getInformation(updateInformation)
        return updateInformation
        
class LocalField(Measurement):
    
    def __init__(self, structure:Structure,Bloc_range,T:float = 0.5,tipPos:int = 0,u:float=1,eta:float=0,
                 G:float=0.03e-6,b0:float=0,Gs:float=1e-6):
        """ Initializes the Measurement class that can be used to calculate the lifetime under influence of 
        a local magnetic field.
        Parameters:
            structure : Structure = the Structure that is measured
            Bloc_range : array = array with local magnetic fields (T)
            T : float = temperature at which the measurement is done (K)
            tipPos : int = atom position of the tip
            u : float = model parameter that stands for elastic tunneling
            eta : float = polarization in the tip along the easy axis
            G : float = tunnel conductance between the tip and the substrate (S)
            b0 : float = part of the electrons that does not interact with the atom
            Gs : float = Surface conductance a measure of the interaction between the atom and the substrate (S)
        """
        super().__init__(structure, T, tipPos, u, eta, G, b0,Gs)
        self.Bloc_range = Bloc_range

        
    def calcT1(self,AntiError = None):
        """ Method that calculates state relaxation time T1 based on the rates from atoms traveling between
        the substrate and the substrate.
        Parameters:
            AntiError = prevents an error if called with parameter
        Returns:
            T1
        """
        
        #Calculation of different rates to the surfes for each state
        _,_,r_ss = self.calcRates()
        #Calculation of the life times of the states
        rates = r_ss.copy()
        #np.fill_diagonal(rates, 0)
        T1 = rates[0,1]**-1
        return T1
        
    def calcT1FullRates(self,V_bias):
        """ Method that calculates state relaxation time T1 based on all rates. 
        Parameters:
            V_bias = the bias at whitch the rates have to be calculated
        Returns:
            T1
        """
        #Loading energies and rates matrices
        r_ts,r_st,r_ss = self.calcRates(V_bias)
        #Calculation of the life times of the states
        rates = r_ss + r_ts + r_st
        #np.fill_diagonal(rates, 0)
        T1 = rates[0.1]**-1
        return T1
    
    def calcNeelProb(self,V_bias = None):
        """ Calculates the probability of each of the Néel states in the ground state and T1 for the first
        excited state.
        Parameters:
            V_bias = the bias at whitch the rates have to be calculated
        Returns:
            Probability of both the Néel states in the ground state
            T1
        """
        calcT1 = self.calcT1FullRates if V_bias is not None else self.calcT1
        
        Htot = self.structure.H
        g0,g1 = self.structure.vectorNeelstate()
        
        T1 = np.zeros((len(self.Bloc_range)))

        NeelProb = np.zeros((2,len(self.Bloc_range)))
        H2 = -self.structure.atoms[self.tipPos].g*muB*self.structure.SList[self.tipPos][2]
        for i,locB in enumerate(self.Bloc_range):
            self.resetEnergy()
            self.structure.resetEnergy()
            H3 = locB*H2
            self.structure.H = Htot + H3
            w,v = SSLA.eigsh(self.structure.H,k = 1, which = 'SA')
            NeelProb[0,i] = np.abs(g0@v[:,0])**2
            NeelProb[1,i] = np.abs(g1@v[:,0])**2
            T1[i] = calcT1(V_bias)
            
        return NeelProb,T1
        
    def getInformation(self,currentInformation):
        """ Method called by Measure form Structure calculates Neelporp and T1 for all local fields given
        """
        keys = currentInformation.keys()
        updateInformation = {}
        newB_range = []
        for Bloc in self.Bloc_range:
            if Bloc not in keys:
                print('+1')
                newB_range.append(Bloc)
            else:
                updateInformation[Bloc] = currentInformation[Bloc] 
        
        if len(newB_range) != 0 :     
            self.Bloc_range = np.array(newB_range)
            Neelprob, T1 = self.calcNeelProb()
            for i,Bloc in enumerate(newB_range):
                print('-1')
                updateInformation[Bloc] = (Neelprob[i],T1[i])
                
        updateInformation = super().getInformation(updateInformation)
        return updateInformation
    
class TemperatureVariation(Measurement):
    
    def __init__(self, structure:Structure,T_range,tipPos:int = 0,u:float=1,eta:float=0,
                 G:float=0.03e-6,b0:float=0,Gs:float=1e-6):
        """ Initializes the Measurement class that can be used to calculate the temperature-dependent 
        lifetime.
        Parameters:
            structure : Structure = the Structure that is measured
            T_range : array = array with all temperatures where the lifetime is calculated for (K)
            tipPos : int = atom position of the tip
            u : float = model parameter that stands for elastic tunneling
            eta : float = polarization in the tip along the easy axis
            G : float = tunnel conductance between the tip and the substrate (S)
            b0 : float = part of the electrons that does not interact with the atom
            Gs : float = Surface conductance a measure of the interaction between the atom and the substrate (S)
        """
        super().__init__(structure, T_range[0], tipPos, u, eta, G, b0,Gs)
        
        self.T_range = T_range
        
    def calcLifeTimes(self):
        """ Caclulates the lifetime of the states for all temperatures when only taking r_ss into account.
        """
        #Loading energies and P matrices
        P_ts,P_st,P_ss = self.calcPmat()
        w = self.structure.getEnergies()
        print(w.shape)
        wx,wy = np.meshgrid(w,w)
        wv = wy - wx
        self.P_ss = P_ss
        #perpare room to save lifetimes
        lifeTimes = np.zeros((len(self.T_range),P_ss.shape[0]))
        #Loop over the Temperatures in T_range
        for i,T_i in enumerate(self.T_range):
            beta_i = 1/(kb*T_i)
            #Calculation of different rates to the surfes for each state
            r_ss = fermiIntAnalitical(0,wv,beta_i)*P_ss*self.Gs/e_charge**2
            #Calculation of the life times of the states
            #rates = r_ss.copy()
            np.fill_diagonal(r_ss, 0)
            lifeTimes[i,:] = ((np.sum(r_ss,axis=0))**-1)
        
        print('Done Lifetime')
        self.lifeTimes = lifeTimes
        
    def calcLifeTimesFullRates(self,V_bias):
        """ Caclulates the lifetime of the states for all temperatures when taking all rates into account.
        Parameters:
            V_bias : float = the bias applied on the tip
        returns:
            The calculated lifetime for all the given temperatures
        """
        #Loading energies and P matrices
        P_ts,P_st,P_ss = self.calcPmat()
        w = self.structure.getEnergies()
        wx,wy = np.meshgrid(w,w)
        wv = wy - wx
        self.P_ss = P_ss
        #perpare room to save lifetimes
        lifeTimes = np.zeros((len(self.T_range),P_ss.shape[0]))
        #Loop over the Temperatures in T_range
        for i,T_i in enumerate(self.T_range):
            beta_i = 1/(kb*T_i)
            #Calculation of different rates to the surfes for each state
            r_ts = fermiIntAnalitical(+V_bias*e_charge,wv,beta_i)*P_ts*self.Gac/e_charge**2
            r_st = fermiIntAnalitical(-V_bias*e_charge,wv,beta_i)*P_st*self.Gac/e_charge**2
            r_ss = fermiIntAnalitical(0,wv,beta_i)*P_ss*self.Gs/e_charge**2
            #Calculation of the life times of the states
            rates = np.abs(r_ss) + np.abs(r_ts) + np.abs(r_st)
            np.fill_diagonal(rates, 0)
            lifeTimes[i,:] = ((np.sum(rates,axis=0))**-1)
            #lifeTimes[i,1] = ((np.sum(rates,axis=0))**-1)
        
        return lifeTimes
    
    def getInformation(self,currentInformation):
        """ Method called by Measure from Structure returns the lifetimes for give T where only r_ss is 
        taken into account.
        """
        keys = currentInformation.keys()
        updateInformation = {}
        newT_range = []
        for T_i in self.T_range:
            if T_i not in keys:
                newT_range.append(T_i)
            else:
                updateInformation[T_i] = currentInformation[T_i] 
        
        if len(newT_range) != 0 :     
            self.T_range = np.array(newT_range)
            lifeTimes = self.getLifeTimes()
            for i,T_i in enumerate(newT_range):
                updateInformation[T_i] = (lifeTimes[i])
                
        updateInformation = super().getInformation(updateInformation)
        return updateInformation
        
class LifeTimeSpinStates(Measurement):     
    
    def __init__(self, structure:Structure,T:float = 0.5,tipPos:int = 0,u:float=1,eta:float=0,
                 G:float=0.03e-6,b0:float=0,Gs:float=1e-6):
        """ Initializes a Measurement class that calculates rates and lifetime from the measurement basis
        instead of the eigenstate basis.
        Parameters:
            structure : Structure = the Structure that is measured
            T : float = temperatue (K)
            tipPos : int = atom position of the tip
            u : float = model parameter that stands for elastic tunneling
            eta : float = polarization in the tip along the easy axis
            G : float = tunnel conductance between the tip and the substrate (S)
            b0 : float = part of the electrons that does not interact with the atom
            Gs : float = Surface conductance a measure of the interaction between the atom and the substrate (S)
        """
        super().__init__(structure, T, tipPos, u, eta, G, b0,Gs)
     
    def calcRates(self,V_bias : float = 0):
        """ Method that calculates the rates for the spin states, not the eigenstates.
        Parameters:
            V_bias : float = Voltage applied to tip
        Returns:
            Returns the calculated rates
        """
        v = self.structure.getEigenStates()
        #Loading energies and P matrices
        P_ts,P_st,P_ss = self.calcPmat()
        
        print(P_ss.shape)
        
        #transforming P from eigenstate basis to spin state basis
        P_ts = np.abs(v)**2 @ P_ts @ np.abs(v.T)**2
        P_st = np.abs(v)**2 @ P_st @ np.abs(v.T)**2
        P_ss = np.abs(v)**2 @ P_ss @ np.abs(v.T)**2
        
        #Take the energies and calculate all the differences
        w = self.structure.getStateEnergies()
        wx,wy = np.meshgrid(w,w)
        wv = wy - wx
        #Calculation of different rates by given bias
        r_ts = fermiIntAnalitical(+V_bias*e_charge,wv,self.beta)*P_ts*self.Gac/e_charge**2
        r_st = fermiIntAnalitical(-V_bias*e_charge,wv,self.beta)*P_st*self.Gac/e_charge**2
        r_ss = fermiIntAnalitical(0,wv,self.beta)*P_ss*self.Gs/e_charge**2
        
        return r_ts,r_st,r_ss
    
    def calcLifeTimes(self):
        """ Method that calculates the lifetime when only taking r_ss into account.
        """
        #Loading rates
        _,_,r_ss = self.calcRates()
        #Make sure rates to same state are taken into account
        np.fill_diagonal(r_ss, 0)
        #Save the calculated lifetime for the states
        self.lifeTimes = ((np.sum(r_ss,axis=0))**-1)
        
    def calcLifeTimesFullRates(self,V_bias):
        """ Method that calculates the lifetime when taking all rates into account.
        Parameters:
            V_bias : float = Voltage applied on the tip
        Returns:
            The calculated lifetime for all spin states
        """
        #Loading in all rates
        r_ts,r_st,r_ss = self.calcRates(V_bias)
        #Add all rates to the total rates
        rates = r_ts + r_st + r_ss
        #Make sure rates to same state are taken into account
        np.fill_diagonal(rates, 0)
        lifeTimes = ((np.sum(rates,axis=0))**-1)
        return lifeTimes
        
        
    def calcLifeTimesTemp(self,T_range):
        """ Method that calculates the lifetime for all temperature in T_range when only taking r_ss
        into account.
        Parameters:
            T_range : array = Array with temperatures for which the lifetimes must be calculated.
        Returns:
            The calculated lifetime for all spin states
        """
        #Loading energies and P matrices
        _,_,P_ss = self.calcPmat()
        w = self.structure.getStateEnergies()
        wx,wy = np.meshgrid(w,w)
        wv = wy - wx
        self.P_ss = P_ss
        #perpare room to save lifetimes
        lifeTimes = np.zeros((len(T_range),P_ss.shape[0]))
        #Loop over the Temperatures in T_range
        for i,T_i in enumerate(T_range):
            beta_i = 1/(kb*T_i)
            #Calculation of different rates to the surfes for each state
            r_ss = fermiIntAnalitical(0,wv,beta_i)*P_ss*self.Gs/e_charge**2
            #Calculation of the life times of the states
            #rates = r_ss.copy()
            np.fill_diagonal(r_ss, 0)
            lifeTimes[i,:] = ((np.sum(r_ss,axis=0))**-1)
        
        return lifeTimes
    
    def calcLifeTimesTempFullRate(self,T_range,V_bias):
        """ Method that calculates the lifetime for all temperature in T_range when taking all rates
        into account.
        Parameters:
            T_range : array = Array with temperatures for which the lifetimes must be calculated.
            V_bias : float = Voltage applied to the tip.
        Returns:
            The calculated lifetime for all spin states.
        """
        v = self.structure.getEigenStates()
        indexA, indexB = self.structure.indexNeelstate()
        #Loading energies and P matrices
        P_ts,P_st,P_ss = self.calcPmat()
        
        print(P_ss.shape)
        #transforming P from eigenstate basis to spin state basis
        P_ts = np.abs(v)**2 @ P_ts @ np.abs(v.T)**2
        P_st = np.abs(v)**2 @ P_st @ np.abs(v.T)**2
        P_ss = np.abs(v)**2 @ P_ss @ np.abs(v.T)**2
        
        P = P_ts + P_st + P_ss
        P_norm = LA.norm(P,axis = 1,keepdims = True)
        
        P_ts /= P_norm
        P_st /= P_norm
        P_ss /= P_norm
        
        print(P_ss.shape)
        #Take the energies and calculate all the differences
        w = self.structure.getStateEnergies()
        #print(w)
        wx,wy = np.meshgrid(w,w)
        wv = wy - wx
        print('u oke?')
        #perpare room to save lifetimes
        lifeTimes = np.zeros((len(T_range),2))
        #Loop over the Temperatures in T_range
        for i,T_i in enumerate(T_range):
            beta_i = 1/(kb*T_i)
            #Calculation of different rates to the surfes for each state
            #Calculation of different rates by given bias
            r_ts = fermiIntAnalitical(+V_bias*e_charge,wv,beta_i)*P_ts*self.Gac/e_charge**2
            r_st = fermiIntAnalitical(-V_bias*e_charge,wv,beta_i)*P_st*self.Gac/e_charge**2
            r_ss = fermiIntAnalitical(0,wv,beta_i)*P_ss*self.Gs/e_charge**2
            #Calculation of the life times of the states
            rates = r_ts + r_st + r_ss
            """
            plt.figure()
            plt.imshow(np.log(rates),origin = 'lower')
            plt.colorbar()
            plt.show()
            """
            
            np.fill_diagonal(rates, 0)
            lifeTimes[i,:] = (np.sum(rates[:,[indexA,indexB]],axis=0))**-1
        
        return lifeTimes
    
      
"""
________________________________________________________________________________________________________
"""
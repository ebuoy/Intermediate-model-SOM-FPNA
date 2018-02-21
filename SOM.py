import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.collections import LineCollection

CARD={"N":0,"E":1,"W":2,"S":3, "n":8}

def distquad(x,y):
    n=np.shape(x)[0]
    dist=0
    for i in range (n):
        dist+=(x[i]-y[i])**2
    return np.sqrt(dist)

def gauss(d,sig):
    return (np.exp(-((d/sig)**2)/2))/(sig)

   
def indice(C,i,j):
    return CARD[C]+4*j+64*i
    
    
def distmat(n,M):
    dist=-1*np.ones((n**2,n**2))
    P=M
    for p in range(1,n**2):
        P=P*M
        for i0 in range(1,n):
            for j0 in range(1,n):
                for i1 in range(1,n):
                    for j1 in range(1,n):
                        if dist[i0+j0][i1+j1] = -1 and P[indice("n",i0,j0)][indice("n",i1,j1)]!=0:
                            dist[i0+j0][i1+j1]=p
    return dist
class Neurone:
    def __init__(self, i, j, cote,data,connections):
        self._i=i
        self._j=j
        self._x=i/cote
        self._y=j/cote
        self._n=cote**2
        if len(data.shape)==2:
            self._weight=np.max(data)*np.random.random(data.shape[1])
        else:
            self._weight=[[] for i in range (data.shape[0])]
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    self._weight[i].append(np.max(data)*np.random.random(data.shape[2]))
            self._weight=np.array(self._weight)
        self._matC=connections
        
class SOM:
    def __init__(self,n,data,MC):
        
        #Définition des paramètres nécessaires à l'entraînement
        self._eps0=0.01
        self._epsmax=0.1
        
        self._sig0=1/4*10**(-1)
        self._sigmax=np.sqrt(10**(-1))
        
        self._n=int(n)#nombre de neurones choisis par ligne pour mod�liser les données
        self._data=np.array(data)
        
        #Initialisation de la grille
        self._nodes=[[] for i in range(self._n)]
        for i in range (self._n):
            for j in range (self._n):
                self._nodes[i].append(Neurone(i,j,self._n,self._n,self._data,MC[i,j]))
        self._nodes=np.array(self._nodes)
                #La grille est initialisée de manière aléatoire
        
        #Initialisation de la matrice d'adjacence
        self._adj=np.zeros((self._n*self._n*8,self._n*self._n*8))
        
        for i in range(self._n*self._n*8):
            for j in range(self._n*self._n*8):
                for k in CARD.keys():
                    if k=="N":
                        indi=indice("S",i-1,j)
                        for h in CARD.keys():
                            if self._nodes[i,j]._matC[CARD[k],CARD[h]]==1:
                                self._adj[indi][indice(h,i,j)]=1
                    elif k=="E":
                        indi=indice("W",i,j+1)
                        for h in CARD.keys():
                            if self._nodes[i,j]._matC[CARD[k],CARD[h]]==1:
                                self._adj[indi][indice(h,i,j)]=1
                    elif k=="W":
                        indi=indice("E",i,j-1)
                        for h in CARD.keys():
                            if self._nodes[i,j]._matC[CARD[k],CARD[h]]==1:
                                self._adj[indi][indice(h,i,j)]=1
                    elif k=="S":
                        indi=indice("N",i+1,j)
                        for h in CARD.keys():
                            if self._nodes[i,j]._matC[CARD[k],CARD[h]]==1:
                                self._adj[indi][indice(h,i,j)]=1
                    elif k=="n":
                        indi=indice("n",i,j)
                        for h in CARD.keys():
                            if self._nodes[i,j]._matC[CARD[k],CARD[h]]==1:
                                self._adj[indi][indice(h,i,j)]=1
                    
        
    def winner(self,vector,distance=distquad):#Par défaut, la distance utilisée est la distance quadratique
        row=self._n
        column=self._n
        dist=[[]for i in range (row)]
        """dist=np.zeros((row,column))
        
        for i in range (row):
            for j in range (column):
                dist[i,j]=distance(self._nodes[i,j]._weight,vector)"""
        for i in range(row):
            for j in range(column):
                dist[i].append(distance(self._nodes[i,j]._weight,vector))
        dist=np.array(dist)
        min=np.argmin(dist)
        iwin=0
        jwin=min
        while jwin>=0:
            jwin-=column
            iwin+=1
        iwin-=1
        jwin+=column
        return(iwin,jwin)
    
    def train(self,k,nbiter,f=gauss,dist_vect=distquad,dist_win_neur=distquad):
        
        mat_dist=distmat(self._n,self._adj)
        eps=self._eps0+(self._epsmax-self._eps0)*(nbiter - k)/nbiter
        sig=self._sig0+(self._sigmax-self._sig0)*(nbiter - k)/nbiter
        
        #eps=self._eps0*(self._epsmax/self._eps0)**((nbiter-k)/nbiter)
        #sig=self._sig0*(self._sigmax/self._sig0)**((nbiter-k)/nbiter)
        
        #Pour l'apprentissage, le vecteur est choisi au hasard
        coordvect=np.random.randint(np.shape(self._data)[0])
        vector=self._data[coordvect]
        iwin,jwin=self.winner(vector)
        self._nodes[iwin,jwin]._weight+=eps*gauss(distquad(self._nodes[iwin,jwin]._weight,vector),sig)*(vector-self._nodes[iwin,jwin]._weight)
        
        #Les voisins du gagnant subissent aussi les effets du changement
        
        for i in range(self._n):
            for j in range(self._n):
                if i!=iwin or j!=jwin:
                    coeff_dist_win_ij=gauss(mat_dist[indice(iwin,jwin)][indice(i,j)],sig)
                    #Coefficient permettant de déterminer le taux d'apprentissage de tous les voisins
                    self._nodes[i,j]._weight+=coeff_dist_win_ij*eps*(vector-self._nodes[i,j]._weight)
        return coordvect,iwin,jwin
    
    def getmap(self):
        
        map=[[] for i in range(self._n)]
        for i in range(self._n):
            for j in range(self._n):
                map[i].append(self._nodes[i,j]._weight)
        
        return np.array(map)
    
    def getmaplist(self):
        map=[]
        for i in range(self._n):
            for j in range(self._n):
                map.append(self._nodes[i,j]._weight)
        
        return np.array(map)
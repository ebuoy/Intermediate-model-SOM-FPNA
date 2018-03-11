import numpy as np
import random
from Graph import *
CARD={"N":0,"E":1,"W":2,"S":3, "n":4}

def dist_quad(x,y):
    return np.sum((x - y) ** 2)

def gauss(d, sig):
    return np.exp(-((d / sig) ** 2) / 2) / sig


def normalized_gaussian(d, sig):
    return np.exp(-((d / sig) ** 2) / 2)

   
def indice(C,i,j):
    return CARD[C]+8*j+128*i

    
def distmat(n,M):
    dist = -1*np.ones((n**2,n**2))
    P = M
    for p in range(1,n**2):
        P = P.dot(M)
        for i0 in range(1,n):
            for j0 in range(1,n):
                for i1 in range(1,n):
                    for j1 in range(1,n):
                        if dist[i0*n+j0][i1*n+j1] == -1 and P[indice("n",i0,j0)][indice("n",i1,j1)] != 0:
                            dist[i0*n+j0][i1*n+j1] = p
    return dist

class Neurone:
    def __init__(self, i, j, cote,connections): #vraies variables self, i, j, cote,data,connections
        self._i = i
        self._j = j
        self._x = i/cote
        self._y = j/cote
        self.n = cote**2
        """if len(data.shape) == 2:
            self._weight = np.max(data)*np.random.random(data.shape[1])
        else:
            self._weight = [[] for i in range (data.shape[0])]
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    self._weight[i].append(np.max(data)*np.random.random(data.shape[2]))
            self._weight = np.array(self._weight)"""
        self._matC = connections



class SOM:
    def __init__(self, n, MC,  distance=dist_quad): # vraies variables : self, n, data, nbEpoch, MC, distance=dist_quad
        
        # Définition des paramètres nécessaires à l'entraînement
        """self.eps0 = 0.9
        self.epsEnd = 0.01
        self.epsilon = self.eps0
        self.epsilon_stepping = (self.epsEnd - self.eps0) / nbEpoch

        self.sig0 = 0.5
        self.sigEnd = 0.025
        self.sigma = self.sig0
        self.sigma_stepping = (self.sigEnd - self.sig0) / nbEpoch"""
        
        self.n=int(n)#nombre de neurones choisis par ligne pour mod�liser les données
        #self.data=np.array(data)
        
        #Initialisation de la grille
        self.nodes=[[] for i in range(self.n)]
        for i in range (self.n):
            for j in range (self.n):
                self.nodes[i].append(Neurone(i,j,self.n, MC[i][j])) #Attention vraies variables non-test i,j,self.n,self.data,MC[i,j]
            
        self.nodes=np.array(self.nodes)
                        #La grille est initialisée de manière aléatoire



        #Initialisation de la matrice d'adjacence
        # self.adj=np.zeros((8+4*self.n+64*self.n, 8+4*self.n+64*self.n))

        self.global_connections = Graph()
        for i in range(self.n):  # Horizontal
            for j in range(self.n):  # Vertical
                self.global_connections.add_edge(Edge("No"+str(i)+","+str(j), "Si"+str(i)+","+str(j-1), 1))
                self.global_connections.add_edge(Edge("So"+str(i)+","+str(j), "Ni"+str(i)+","+str(j+1), 1))
                self.global_connections.add_edge(Edge("Eo"+str(i)+","+str(j), "Wi"+str(i+1)+","+str(j), 1))
                self.global_connections.add_edge(Edge("Wo"+str(i)+","+str(j), "Ei"+str(i-1)+","+str(j), 1))

        for i in range(self.n):
            for j in range(self.n):
                for k in range(5):
                    for l in range(5):
                        if self.nodes[j, i]._matC[k, l] != 0:
                            inp = SOM.get_index(k, 'i')+str(i)+','+str(j)
                            out = SOM.get_index(l, 'o')+str(i)+','+str(j)
                            e = Edge(inp, out, self.nodes[j, i]._matC[k, l])
                            self.global_connections.add_edge(e)

        self.adj = self.global_connections.get_adjacency_matrix()
        self.MDist = self.global_connections.get_all_shortest_paths()

        """
        for i in range(self.n):
            for j in range(self.n):
                for k in CARD.keys():
                    if k =="N":
                        indi = indice("S",i-1,j)
                        for h in CARD.keys():
                            if self.nodes[i,j]._matC[CARD[k]][CARD[h]] == 1:
                                self.adj[indi][indice(h,i,j)] = 1
                                
                    elif k == "E":
                        indi = indice("W",i,j+1)
                        for h in CARD.keys():
                            if self.nodes[i,j]._matC[CARD[k]][CARD[h]] == 1:
                                self.adj[indi][indice(h,i,j)] = 1
                                
                    elif k == "W":
                        indi = indice("E",i,j-1)
                        for h in CARD.keys():
                            if self.nodes[i,j]._matC[CARD[k]][CARD[h]] == 1:
                                self.adj[indi][indice(h,i,j)] = 1
                                
                    elif k == "S":
                        indi = indice("N",i+1,j)
                        for h in CARD.keys():
                            if self.nodes[i,j]._matC[CARD[k]][CARD[h]] == 1:
                                self.adj[indi][indice(h,i,j)] = 1
                                
                    elif k == "n":
                        for h in CARD.keys():
                            if self.nodes[i,j]._matC[4][CARD[h]] == 1:
                                self.adj[indi][indice(h,i,j)] = 1
        
        self.MDist = distmat(self.n,self.adj)
        """

    def compute_neurons_distance(self):
        neurons_list = []
        for i in range(self.n):
            for j in range(self.n):
                neurons_list.append("n"+str(i)+","+str(j))
        self.neural_dist = self.global_connections.get_shortest_paths(neurons_list)

    @staticmethod
    def get_index(x, y):
        return {
            0: "N"+y,    # North
            1: "E"+y,    # East
            2: "W"+y,    # West
            3: "S"+y,    # South
            4: "n"       # neuron
        }.get(x)

    def winner(self, vector, distance=dist_quad):
        dist = np.empty_like(self.nodes)
        for i in range(self.row):  # Computes the distances between the tested vector and all nodes
            for j in range(self.column):
                dist[i][j] = distance(self.nodes[i, j].weight, vector)
        return np.unravel_index(np.argmin(dist, axis=None), dist.shape)  # Returning the Best Matching Unit's index.

    def train(self, k, epochTime, f=normalized_gaussian, distance=dist_quad):
        if k % epochTime == 0:
            self.epsilon += self.epsilon_stepping
            self.sigma += self.sigma_stepping
            self.generate_random_list()

        # The training vector is chosen randomly
        vector_coordinates = self.unique_random_vector()
        vector = self.data[vector_coordinates]

        # Getting the Best matching unit
        bmu = self.winner(vector, distance)
        self.updating_weights(bmu, vector, f)

        return vector_coordinates, bmu[0], bmu[1]

    def updating_weights(self, bmu, vector, f=normalized_gaussian):
        # Updating weights of all nodes
        for i in range(self.row):
            for j in range(self.column):
                dist = np.sqrt(self.MDist[i, bmu[0]] + self.MDist[j, bmu[1]])/np.sqrt(2)  # Normalizing the distances
                self.nodes[i, j].weight += f(dist, self.sigma)*self.epsilon*(vector-self.nodes[i, j].weight)

    def fully_random_vector(self):
        return np.random.randint(np.shape(self.data)[0])

    def unique_random_vector(self):
        return self.vector_list.pop(0)

    def generate_random_list(self):
        self.vector_list = list(range(len(self.data)))
        random.shuffle(self.vector_list)
    
    def getmap(self):
        map=[[] for i in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                map[i].append(self.nodes[i,j]._weight)
        
        return np.array(map)
    
    def getmaplist(self):
        map=[]
        for i in range(self.n):
            for j in range(self.n):
                map.append(self.nodes[i,j]._weight)
        
        return np.array(map)
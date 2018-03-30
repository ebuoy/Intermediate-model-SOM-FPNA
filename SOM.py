import numpy as np
import random
from Graph import *


def dist_quad(x, y):
    return np.sum((x - y) ** 2)


def gauss(d, sig):
    return np.exp(-((d / sig) ** 2) / 2) / sig


def normalized_gaussian(d, sig):
    return np.exp(-((d / sig) ** 2) / 2)


class Neurone:
    def __init__(self, i, j, cote, data, connections):
        self.i = i
        self.j = j
        self.x = i/cote
        self.y = j/cote
        self.n = cote**2
        self.matC = connections
        self.set_random_weights(data)

    def set_random_weights(self, data):
        if len(data.shape) == 2:
            self.weight = np.max(data)*np.random.random(data.shape[1])
        else:
            self.weight = [[] for i in range (data.shape[0])]
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    self.weight[i].append(np.random.random(data.shape[2]))
            self.weight = np.array(self.weight)


class SOM:
    def __init__(self, n, data, nbEpoch, MC,  distance=dist_quad): # vraies variables : self, n, data, nbEpoch, MC, distance=dist_quad
        
        # Définition des paramètres nécessaires à l'entraînement
        self.eps0 = 0.9
        self.epsEnd = 0.01
        self.epsilon = self.eps0
        self.epsilon_stepping = (self.epsEnd - self.eps0) / nbEpoch

        self.sig0 = 0.5
        self.sigEnd = 0.025
        self.sigma = self.sig0
        self.sigma_stepping = (self.sigEnd - self.sig0) / nbEpoch
        
        self.n = int(n)  # nombre de neurones choisis par ligne pour mod�liser les données
        self.data = np.array(data)/255
        
        # Initialisation de la grille
        self.nodes = [[] for i in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                self.nodes[i].append(Neurone(i, j, self.n, self.data, MC[i][j]))
        self.nodes = np.array(self.nodes)

        # Calculating adjacency matrix
        self.global_connections = Graph()
        for i in range(self.n):  # Horizontal
            for j in range(self.n):  # Vertical
                if j != 0:
                    self.global_connections.add_edge(Edge("No"+str(i)+","+str(j), "Si"+str(i)+","+str(j-1), 0))
                if j != self.n-1:
                    self.global_connections.add_edge(Edge("So"+str(i)+","+str(j), "Ni"+str(i)+","+str(j+1), 0))
                if i != self.n-1:
                    self.global_connections.add_edge(Edge("Eo"+str(i)+","+str(j), "Wi"+str(i+1)+","+str(j), 0))
                if i != 0:
                    self.global_connections.add_edge(Edge("Wo"+str(i)+","+str(j), "Ei"+str(i-1)+","+str(j), 0))

        for i in range(self.n):
            for j in range(self.n):
                for k in range(5):
                    for l in range(5):
                        if self.nodes[j, i].matC[k, l] != 0:
                            inp = SOM.get_index(k, 'i')+str(i)+','+str(j)
                            out = SOM.get_index(l, 'o')+str(i)+','+str(j)
                            e = Edge(inp, out, SOM.neurons_only_weight(k, l))
                            self.global_connections.add_edge(e)

        self.adj = self.global_connections.get_adjacency_matrix()
        self.global_connections.extract_neurons_graph()
        self.neural_graph = self.global_connections.extract_neurons_graph()
        self.neural_graph.print()
        print(self.neural_graph.to_string())
        self.compute_neurons_distance()

    def compute_neurons_distance(self):
        self.neural_dist = self.neural_graph.get_all_shortest_paths()
        print(self.neural_dist)
        self.MDist = np.array(self.neural_dist)

        maximum = -1
        for i in range(neuron_nbr):
            for j in range(neuron_nbr):
                if self.MDist[i][j] != np.Infinity and self.MDist[i][j] > maximum:
                    maximum = self.MDist[i][j]
        self.MDist = np.divide(self.MDist, maximum)  # Normalizing the distances

    def remove_edges(self, v1, v2):
        inp = "n"+str(v1[0])+','+str(v1[1])
        out = "n"+str(v2[0])+','+str(v2[1])
        self.neural_graph.remove_edge(inp, out)
        self.neural_graph.remove_edge(out, inp)
        self.compute_neurons_distance()

    @staticmethod
    def get_index(x, y):
        return {
            0: "N"+y,    # North
            1: "E"+y,    # East
            2: "W"+y,    # West
            3: "S"+y,    # South
            4: "n"       # neuron
        }.get(x)

    @staticmethod
    def uniform_weight(k, l):
        return 1

    @staticmethod
    def neurons_only_weight(k, l):
        if k == 4 or l == 4:
            return 0.5
        return 0

    def winner(self, vector, distance=dist_quad):
        dist = np.empty_like(self.nodes)
        for i in range(self.n):  # Computes the distances between the tested vector and all nodes
            for j in range(self.n):
                dist[i][j] = distance(self.nodes[i, j].weight, vector)
        return np.unravel_index(np.argmin(dist, axis=None), dist.shape)  # Returning the Best Matching Unit's index.

    def train(self, k, epoch_time, f=normalized_gaussian, distance=dist_quad):
        if k % epoch_time == 0:
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
        for i in range(self.n):
            for j in range(self.n):
                dist = self.MDist[bmu[1]*self.n+bmu[0], j*self.n+i]
                if dist != np.Infinity:
                    self.nodes[i, j].weight += f(dist, self.sigma)*self.epsilon*(vector-self.nodes[i, j].weight)

    def fully_random_vector(self):
        return np.random.randint(np.shape(self.data)[0])

    def unique_random_vector(self):
        return self.vector_list.pop(0)

    def generate_random_list(self):
        self.vector_list = list(range(len(self.data)))
        random.shuffle(self.vector_list)
    
    def getmap(self):
        map = [[] for i in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                map[i].append(self.nodes[i, j].weight)
        return np.array(map) * 255
    
    def getmaplist(self):
        map = []
        for i in range(self.n):
            for j in range(self.n):
                map.append(np.array(self.nodes[i, j].weight) * 255)
        return map

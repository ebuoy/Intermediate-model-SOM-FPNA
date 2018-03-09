import numpy as np


class Edge:
    def __init__(self, in_vertex, out_vertex, weight):
        self.in_vertex = in_vertex
        self.out_vertex = out_vertex
        self.weight = weight


class Graph:
    def __init__(self, vertices_number):
        self.vertices_number = vertices_number
        self.edges_list = []

    def add_edge(self, edge):
        self.edges_list.append(edge)

    def get_adjacency_matrix(self):
        adjacency_matrix = np.full((self.vertices_number, self.vertices_number), np.Infinity)
        for e in self.edges_list :
            adjacency_matrix[e.in_vertex, e.out_vertex] = e.weight
        return adjacency_matrix

    def get_all_shortest_paths(self):
        # We are using the Floyd-Warshall algorithm
        adj_matrix = self.get_adjacency_matrix()
        for k in range(self.vertices_number):
            for i in range(self.vertices_number):
                for j in range(self.vertices_number):
                    adj_matrix[i, j] = np.minimum(adj_matrix[i, j], adj_matrix[i, k] + adj_matrix[k, j])
        return adj_matrix
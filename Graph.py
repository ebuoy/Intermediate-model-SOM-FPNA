import numpy as np


class Edge:
    def __init__(self, in_vertex, out_vertex, weight):
        self.in_vertex = in_vertex
        self.out_vertex = out_vertex
        self.weight = weight

    def to_string(self):
        return self.in_vertex+"--"+str(self.weight)+"->"+self.out_vertex

class Graph:
    def __init__(self):
        self.edges_list = []
        self.vertex_list = {}

    def add_edge(self, edge):
        if edge.in_vertex not in self.vertex_list:
            self.vertex_list[edge.in_vertex] = len(self.vertex_list)
        if edge.out_vertex not in self.vertex_list:
            self.vertex_list[edge.out_vertex] = len(self.vertex_list)
        self.edges_list.append(edge)

    def get_adjacency_matrix(self):
        adjacency_matrix = np.full((len(self.vertex_list), len(self.vertex_list)), np.Infinity)
        for e in self.edges_list:
            adjacency_matrix[self.vertex_list[e.in_vertex], self.vertex_list[e.out_vertex]] = e.weight
        return adjacency_matrix

    def get_all_shortest_paths(self):
        # We are using the Floyd-Warshall algorithm
        dist_matrix = self.get_adjacency_matrix()
        for k in range(len(self.vertex_list)):
            for i in range(len(self.vertex_list)):
                for j in range(len(self.vertex_list)):
                    dist_matrix[i, j] = np.minimum(dist_matrix[i, j], dist_matrix[i, k] + dist_matrix[k, j])
        return dist_matrix

    def get_shortest_paths(self, chosen_vertex):
        dist_matrix = self.get_all_shortest_paths()
        chosen_dist = np.full((len(chosen_vertex), len(chosen_vertex)), np.Infinity)
        for i in range(len(chosen_vertex)):
            for j in range(len(chosen_vertex)):
                chosen_dist[i, j] = dist_matrix[self.vertex_list[chosen_vertex[i]], self.vertex_list[chosen_vertex[j]]]
        return chosen_dist

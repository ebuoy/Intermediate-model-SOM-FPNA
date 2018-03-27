import numpy as np
from scipy.sparse.csgraph import floyd_warshall
from heapq import *


class Edge:
    def __init__(self, in_vertex, out_vertex, weight):
        self.in_vertex = in_vertex
        self.out_vertex = out_vertex
        self.weight = weight

    def to_string(self):
        return self.in_vertex+"--"+str(self.weight)+"-->"+self.out_vertex

    def __lt__(self, other):
        return self.weight < other.weight


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
            if e.weight == 0:
                adjacency_matrix[self.vertex_list[e.in_vertex], self.vertex_list[e.out_vertex]] = 0.01
            else:
                adjacency_matrix[self.vertex_list[e.in_vertex], self.vertex_list[e.out_vertex]] = e.weight
        for v in self.vertex_list:
            adjacency_matrix[self.vertex_list[v], self.vertex_list[v]] = 0
        return adjacency_matrix

    def get_all_shortest_paths(self):
        return floyd_warshall(self.get_adjacency_matrix())

    def get_shortest_paths(self, chosen_vertex):
        dist_matrix = self.get_all_shortest_paths()
        chosen_dist = np.full((len(chosen_vertex), len(chosen_vertex)), np.Infinity)
        for i in range(len(chosen_vertex)):
            for j in range(len(chosen_vertex)):
                chosen_dist[i, j] = dist_matrix[self.vertex_list[chosen_vertex[i]], self.vertex_list[chosen_vertex[j]]]
        return chosen_dist

    def extract_neurons_graph(self):
        new_edges_list = []
        link_edges_list = []
        finished = []
        for e in self.edges_list:
            if e.in_vertex[0] == 'n':
                new_edges_list.append(e)
            else:
                link_edges_list.append(e)
        hash_map = [[] for i in range(len(self.vertex_list))]
        for e in link_edges_list:
            hash_map[self.vertex_list[e.in_vertex]].append(e)
        while new_edges_list:
            current = new_edges_list.pop()
            for i in hash_map[self.vertex_list[current.out_vertex]]:
                new = Edge(current.in_vertex, i.out_vertex, current.weight+i.weight)
                if new.out_vertex[0] == 'n':
                    finished.append(new)
                else:
                    new_edges_list.append(new)
        g = Graph()
        for e in finished:
            g.add_edge(e)
        return g

    def test(self):
        global open_list, hash_map, current, dist
        open_list = []
        hash_map = [[] for i in range(len(self.vertex_list))]
        dist = self.get_adjacency_matrix()
        for e in self.edges_list:
            hash_map[self.vertex_list[e.in_vertex]].append(e)
            heappush(open_list, e)

        while open_list:
            current = heappop(open_list)
            self.update()

        return dist

    def update(self):
        if current.weight == dist[self.vertex_list[current.in_vertex], self.vertex_list[current.out_vertex]]:
            for e in hash_map[self.vertex_list[current.out_vertex]]:
                w = current.weight + e.weight
                if w < dist[self.vertex_list[current.in_vertex], self.vertex_list[e.out_vertex]]:
                    dist[self.vertex_list[current.in_vertex], self.vertex_list[e.out_vertex]] = w
                    new = Edge(current.in_vertex, e.out_vertex, w)
                    heappush(open_list, new)

    def to_string(self):
        res = "Vertices : "+str(self.vertex_list)+"\nEdges :\n"
        for e in self.edges_list:
            res += e.to_string+"\n"
        return res

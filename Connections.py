from Images import *
from Connections_Models import *


def kohonen():
    connexion_matrix = np.empty((neuron_nbr, neuron_nbr, 5, 5))
    for i in range(neuron_nbr):
        for j in range(neuron_nbr):
            connexion_matrix[i, j] = kohonen_matrix
    return connexion_matrix


def small_worlds():
    connexion_matrix = np.empty((neuron_nbr, neuron_nbr, 5, 5))
    pattern = [[top_left, top_highway, top_right],
               [left_highway, kohonen_matrix, right_highway],
               [bottom_left, bottom_highway, bottom_right]]
    for i in range(neuron_nbr):
        for j in range(neuron_nbr):
            connexion_matrix[i, j] = pattern[i % 3][j % 3]
    return connexion_matrix


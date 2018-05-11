import numpy as np
np.set_printoptions(threshold=np.inf)  # Used to print the data completely

# Images
pictures_dim = (4, 4)
output_path = "./results/compression/"

# SOM variables
neuron_nbr = 8
epoch_nbr = 40
epsilon_start = 0.9
epsilon_end = 0.1
sigma_start = 0.5
sigma_end = 0.025

# PSOM variant
psom = False
omega = 10**(-8)

# Genetic Optimisation
range_epoch_nbr = (20, 100)
range_epsilon_start = (0.1, 1)
range_epsilon_end = (0.001, 1)
range_sigma_start = (0.1, 1)
range_sigma_end = (0.001, 1)

probability_neural_link = 0.5
probability_link = 0.2

probability_mutation = 0.2
mutation_value = 0.2
nb_individuals = 4
nb_generations = 2
elite_proportion = 0.4

# Logs
log_graphs = False
log_gaussian_vector = False
log_execution = False

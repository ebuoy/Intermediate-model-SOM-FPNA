import numpy as np
np.set_printoptions(threshold=np.inf)  # Used to print the data completely

# Images
pictures_dim = (9, 9)
output_path = "./results/final_test/"
input_path = "./image/limited_test/"

# SOM variables
neuron_nbr = 12
epoch_nbr = 50
epsilon_start = 0.6
epsilon_end = 0.05
sigma_start = 0.5
sigma_end = 0.001

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

probability_mutation = 0.1
mutation_value = 0.1
nb_individuals = 20
nb_generations = 50
elite_proportion = 0.4

# Logs
log_graphs = False
log_gaussian_vector = False
log_execution = False

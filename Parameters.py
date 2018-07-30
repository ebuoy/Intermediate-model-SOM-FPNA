import numpy as np
np.set_printoptions(threshold=np.inf)  # Used to print the data completely

# Images
pictures_dim = (3, 3)
output_path = "./results/dynamic/"
input_path = "./image/limited_test/"

# SOM variables
neuron_nbr = 6
epoch_nbr = 20
epsilon_start = 0.6
epsilon_end = 0.05
sigma_start = 0.5
sigma_end = 0.001

# PSOM variant
psom = False
omega = 3*10**(-7)

# Star center change
dsom = False
switch_threshold = 1.5
range_threshold = (0.5, 5)

# DSOM
elasticity = 1.75
dsom_epsilon = 0.5

# GNG SOM
epsilon_b = 0.4
epsilon_n = 0.2

# PC SOM
alpha_start = 0.1
alpha_end = 0.05
eta_start = 0.01
eta_end = 0.001


# Genetic Optimisation
range_epoch_nbr = (50, 50)
range_epsilon_start = (0.01, 1)
range_epsilon_end = (0.0001, 1)
range_sigma_start = (0.01, 1)
range_sigma_end = (0.0001, 1)

probability_neural_link = 0.5
probability_link = 0.2

probability_mutation = 0.1
mutation_value = 0.1
nb_individuals = 20
nb_generations = 50
elite_proportion = 0.4

# Logs
log_graphs = False
log_gaussian_vector = True
log_execution = False

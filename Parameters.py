import numpy as np
np.set_printoptions(threshold=np.inf)  # Used to print the data completely

# Images
pictures_dim = (4, 4)
output_path = "./results/"

# SOM variables
neuron_nbr = 12
epoch_nbr = 20
epsilon_start = 0.9
epsilon_end = 0.1
sigma_start = 0.5
sigma_end = 0.025

# PSOM variant
psom = False
omega = 10**(-8)

# Logs
log_graphs = False
log_gaussian_vector = False

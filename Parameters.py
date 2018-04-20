import numpy as np
np.set_printoptions(threshold=np.inf)  # Used to print the data completely

# Images
pictures_dim = (3, 3)
output_path = "./results/deep/"

# SOM variables
neuron_nbr = 4
epoch_nbr = 300
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

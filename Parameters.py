import numpy as np
np.set_printoptions(threshold=np.inf)  # Used to print the data completely

# Images
pictures_dim = (10, 10)
output_path = "./results/"

# SOM variables
neuron_nbr = 10
epoch_nbr = 100
epsilon_start = 0.9
epsilon_end = 0.1
sigma_start = 0.5
sigma_end = 0.025

# PSOM variant
psom = True
omega = 10**(-8)



from Images import *
from SOM import *
np.set_printoptions(threshold=np.inf)  # used to display numpy arrays in full
global kohonen_matrix
kohonen_matrix = [[0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 1],
                  [1, 1, 1, 1, 0]]


def display_som(som_list):
    nothing = 0
    #TODO : implémenter


def compute_mean_error(datacomp, datamat, SOMList):
    error = np.zeros(len(datacomp))
    for i in range(len(datacomp)):
        error[i] = np.mean(np.abs(datamat[i] - SOMList[datacomp[i]]))
    return np.mean(error)


def run():
    img = Dataset("./image/Audrey.png")
    data = img.data

    datacomp = np.zeros(len(data), int)  # datacomp est la liste du numéro du neurone vainqueur pour l'imagette correspondante
    old = datacomp

    nb_epoch = 100
    epoch_time = len(data)
    nb_iter = epoch_time * nb_epoch
    connexion_matrix = np.empty((neuron_nbr, neuron_nbr, 5, 5))
    for i in range(neuron_nbr):
        for j in range(neuron_nbr):
            connexion_matrix[i, j] = kohonen_matrix

    carte = SOM(neuron_nbr, data, nb_epoch, connexion_matrix)

    for i in range(nb_iter):
        vect, iwin, jwin = carte.train(i, epoch_time)
        datacomp[vect] = iwin * neuron_nbr + jwin
        if i % epoch_time == 0:
            print("Epoch : ", i // epoch_time + 1, "/", nb_epoch)
            diff = np.count_nonzero(datacomp - old)
            print("Changed values :", diff)
            print("Mean error : ", compute_mean_error(datacomp, data, carte.getmaplist()))
            old = np.array(datacomp)
    img.compression(carte)
    # display_som(carte.getmap())

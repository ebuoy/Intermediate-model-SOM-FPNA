from Images import *
from SOM import *
from Connections import *
np.set_printoptions(threshold=np.inf)  # used to display numpy arrays in full


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

    nb_epoch = 300
    epoch_time = len(data)
    nb_iter = epoch_time * nb_epoch

    carte = SOM(neuron_nbr, data, nb_epoch, small_worlds())

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


def test_distances():
    connexion_matrix = np.empty((neuron_nbr, neuron_nbr, 5, 5))
    for i in range(neuron_nbr):
        for j in range(neuron_nbr):
            connexion_matrix[i, j] = kohonen_matrix

    SOM(neuron_nbr, data, 1, connexion_matrix)

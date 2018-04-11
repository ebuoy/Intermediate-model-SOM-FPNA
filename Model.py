from Images import *
from SOM import *
from Connections import *


def display_som(som_list):
    px2 = []
    lst2 = ()
    for i in range(neuron_nbr):
        lst = ()
        for j in range(neuron_nbr):
            som_list[i * neuron_nbr + j] = som_list[i * neuron_nbr + j].reshape(pictures_dim)
            lst = lst + (som_list[i * neuron_nbr + j],)
        px2.append(np.concatenate(lst, axis=1))
        lst2 += (px2[i],)
    px = np.concatenate(lst2, axis=0)
    som_image = Image.fromarray(px)
    som_image.show()
    return som_image


def compute_mean_error(datacomp, datamat, SOMList):
    error = np.zeros(len(datacomp))
    for i in range(len(datacomp)):
        error[i] = np.mean(np.abs(datamat[i] - SOMList[datacomp[i]]))
    return np.mean(error)


def run():
    img = Dataset("./image/Audrey.png")
    data = img.data # load_image_folder("./image/")

    datacomp = np.zeros(len(data), int)  # datacomp est la liste du numéro du neurone vainqueur pour l'imagette correspondante
    old = datacomp

    epoch_time = len(data)
    nb_iter = epoch_time * epoch_nbr

    carte = SOM(data, kohonen())

    for i in range(nb_iter):
        vect, iwin, jwin = carte.train(i, epoch_time)
        datacomp[vect] = jwin * neuron_nbr + iwin
        if i % epoch_time == 0:
            print("Epoch : ", i // epoch_time + 1, "/", epoch_nbr)
            diff = np.count_nonzero(datacomp - old)
            print("Changed values :", diff)
            print("Mean error : ", compute_mean_error(datacomp, data, carte.get_som_as_list()))
            old = np.array(datacomp)
            if psom:
                carte.pruning_neighbors()

    carte.neural_graph.print()
    img.compression(carte)
    display_som(carte.get_som_as_list())


from Images import *
from SOM import *
from Connections import *
np.set_printoptions(threshold=np.inf)  # used to display numpy arrays in full

def noLink():
    pix = np.full(pictures_dim,255)
    return pix

def hLink():
    pix = np.full(pictures_dim,255)
    mid = pictures_dim[0]//2
    for j in range(pictures_dim[1]):
        pix[mid][j]=0
    return pix

def vLink():
    pix = np.full(pictures_dim,255)
    mid = pictures_dim[1]//2
    for i in range(pictures_dim[0]):
        pix[i][mid]=0
    return pix

def display_som(som_list,adj):
    #px2 = []
    lst2 = ()
    for i in range(neuron_nbr):
        lst = ()
        for j in range(neuron_nbr):
            som_list[i * neuron_nbr + j] = som_list[i * neuron_nbr + j].reshape(pictures_dim)
            lst = lst + (som_list[i * neuron_nbr + j],)
            if (j<neuron_nbr-1):
                if (adj[i*neuron_nbr+j][i*neuron_nbr+j+1]==0):
                    lst = lst + (noLink(),)
                else:
                    lst = lst + (hLink(),)
        #px2.append(np.concatenate(lst, axis=1))
        lst2 += (np.concatenate(lst, axis=1),)
        lst = ()
        if (i<neuron_nbr-1):
            for j in range(neuron_nbr):
                if (adj[i*neuron_nbr+j][(i+1)*neuron_nbr+j]==0):
                    lst = lst + (noLink(),)
                else:
                    lst = lst + (vLink(),)
                if (j<neuron_nbr-1):
                    lst = lst + (noLink(),)
            #px2.append(np.concatenate(lst, axis=1))
            lst2 += (np.concatenate(lst, axis=1),)
    px = np.concatenate(lst2, axis=0)
    px = np.array(px, 'uint8')

    som_image = Image.fromarray(px)
    #som_image.show()
    return som_image


def compute_mean_error(datacomp, datamat, SOMList):
    error = np.zeros(len(datacomp))
    for i in range(len(datacomp)):
        error[i] = np.mean(np.abs(datamat[i] - SOMList[datacomp[i]]))
    return np.mean(error)


def run():
    img = Dataset("./image/Audrey.png")
    data = img.data
    #data = load_image_folder("./image/")

    datacomp = np.zeros(len(data), int)  # datacomp est la liste du numero du neurone vainqueur pour l'imagette correspondante
    old = np.array(datacomp)
    Pold = np.array(datacomp)
    Pdatacomp = np.array(datacomp)

    nb_epoch = 20
    epoch_time = len(data)
    nb_iter = epoch_time * nb_epoch

    carte = SOM(neuron_nbr, data, nb_epoch, kohonen())
    Pcarte = carte.copy()

    datacomp = carte.winners()
    Pdatacomp = Pcarte.winners()
    print("Initial mean error SOM: ", compute_mean_error(datacomp, data, carte.getmaplist()))
    print("Initial mean error PSOM : ", compute_mean_error(Pdatacomp, data, Pcarte.getmaplist()))
    for i in range(nb_iter):
         # The training vector is chosen randomly
        if i % epoch_time == 0:
             carte.generate_random_list()

        vect = carte.unique_random_vector()
        iwin, jwin = carte.train(i, epoch_time, vect)
        Piwin, Pjwin = Pcarte.train(i, epoch_time, vect)
        #datacomp[vect] = iwin * neuron_nbr + jwin
        #Pdatacomp[vect] = Piwin * neuron_nbr + Pjwin
        if (i+1) % epoch_time == 0:
            print("Epoch : ", (i+1) // epoch_time , "/", nb_epoch)
            datacomp = carte.winners()
            #print(datacomp)
            Pdatacomp = Pcarte.winners()
            diff = np.count_nonzero(datacomp - old)
            Pdiff = np.count_nonzero(Pdatacomp - Pold)
            print("Changed values SOM :", diff)
            print("Changed values PSOM :", Pdiff)
            print("Mean error SOM: ", compute_mean_error(datacomp, data, carte.getmaplist()))
            print("Mean error PSOM : ", compute_mean_error(Pdatacomp, data, Pcarte.getmaplist()))
            old = np.array(datacomp)
            Pold = np.array(Pdatacomp)
            Pcarte.pruning_neighbors()
            # if i // epoch_time % 10 == 0 and i // epoch_time > nb_epoch/2:
            #     carte.cut_close_neighbors()

    img.compression(carte,"comp2.png")
    im1 = display_som(carte.getmaplist(),carte.adj)
    im1.save("./carte2.png")

    img.compression(Pcarte,"Pcomp2.png")
    im2 = display_som(Pcarte.getmaplist(),Pcarte.adj)
    im2.save("./Pcarte2.png")

# Model :
run()

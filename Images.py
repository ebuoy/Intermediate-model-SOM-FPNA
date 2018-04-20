from PIL import Image
from SOM import *
from Parameters import *
import os


class Dataset:
    def __init__(self, path):
        self.nb_pictures = None
        self.data = None
        self.load_image(path)

    def load_image(self, path):
        self.data = []
        im = Image.open(path)
        size = np.flip(im.size, 0)  # For some strange reason the data isn't ordered in the same way as the size says
        px = np.array(im.getdata(), 'uint8')
        if len(px.shape) == 2:  # File has RGB colours
            px = np.hsplit(px, 3)[0]
        px = px.reshape(size)
        self.nb_pictures = np.array(np.divide(size, pictures_dim), dtype=int)
        px = px[0:self.nb_pictures[0] * pictures_dim[0], 0:self.nb_pictures[1] * pictures_dim[1]]  # Cropping the image to make it fit
        px = np.vsplit(px, self.nb_pictures[0])
        for i in px:
            j = np.hsplit(i, self.nb_pictures[1])
            for picture in j:
                self.data.append(picture.flatten())

        print("\n" + path)
        print("Pictures number :", self.nb_pictures)
        if size[0] / pictures_dim[0] != self.nb_pictures[0] or size[0] / pictures_dim[0] != self.nb_pictures[0]:
            print("\tWarning - image size is not divisible by pictures dimensions, the result will be cropped")

    def compression(self, som, name):
        som_map = som.get_som_as_map()
        pixels = []
        winners = []
        for i in range(len(self.data)):
            w = som.winner(self.data[i]/255)
            if w not in winners:
                winners.append(w)
            pixels.append(som_map[w])
        px2 = []
        lst2 = ()
        for i in range(self.nb_pictures[0]):
            lst = ()
            for j in range(self.nb_pictures[1]):
                pixels[i*self.nb_pictures[1]+j] = pixels[i*self.nb_pictures[1]+j].reshape(pictures_dim)
                lst = lst + (pixels[i*self.nb_pictures[1]+j],)
            px2.append(np.concatenate(lst, axis=1))
            lst2 += (px2[i],)
        px = np.concatenate(lst2, axis=0)
        px = np.array(px, 'uint8')

        file = Image.fromarray(px)
        #file.show()
        file.save(output_path+name)

        n = neuron_nbr*neuron_nbr
        print("Used neurons :", len(winners), "/", n, "(", len(winners)/n*100, "%)")

    def save_compressed(self, som, name):
        winners = np.zeros((neuron_nbr, neuron_nbr))
        for i in range(len(self.data)):
            w = som.winner(self.data[i]/255)
            winners[w] += 1
        file = open(output_path+name, "w")
        # file.write(str(som.get_som_as_list()))
        res = ""
        for i in range(neuron_nbr):
            for j in range(neuron_nbr):
                res += str(winners[i, j])+" "
        file.write(res)
        file.close()


def load_image_folder(path):
    files = os.listdir(path)
    data = []
    for f in files:
        data.extend(Dataset(path + f).data)
    return data


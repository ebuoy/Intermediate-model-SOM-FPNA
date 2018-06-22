from Model import *
from tkinter import *
from tkinter.ttk import *
from Simple_Data_Sample import *
from Connections import *
import time

radius = 3
width = 600
height = 600
dataset_size = 2000


def draw_data_point(canvas, x, y, fill_color):
    canvas.create_oval(x-radius, y-radius, x+radius, y+radius, outline=fill_color, fill=fill_color)


def project(x, y):
    return radius+x*(width-radius), radius+y*(height-radius)


class GraphicalSOM:
    def __init__(self):
        self.window = Tk()
        self.window.title("Graphical Self-Organised Map")
        self.canvas = Canvas(self.window, width=width, height=height, bg="ivory")
        self.canvas.grid(row=0, column=0, columnspan=10, rowspan=10, padx=10, pady=10)
        self.SOM = SOM(sierpinski_carpet(dataset_size, 5), star())
        self.epoch_time = len(self.SOM.data)
        self.current_iteration = 0
        self.total_iterations = self.epoch_time * epoch_nbr
        self.running = False

        self.draw_buttons()
        self.draw_data()
        self.draw_map()
        self.draw_metrics()
        self.canvas.update()
        self.window.mainloop()

    def draw_buttons(self):
        self.number = Entry(self.window, text="1")
        self.listbox = Combobox(self.window, state="readonly")
        self.listbox['values'] = ("iterations", "epochs")
        self.listbox.current(0)
        self.step = Button(self.window, text='Step-by-step', command=self.run_once)
        self.run = Button(self.window, text='Run', command=self.run)

        self.number.grid(row=10, column=0)
        self.listbox.grid(row=10, column=1)
        self.step.grid(row=10, column=2)
        self.run.grid(row=10, column=3)

    def draw_data(self):
        data = self.SOM.data
        for element in data:
            x, y = project(element[0], element[1])
            draw_data_point(self.canvas, x, y, "red")

    def draw_map(self):
        positions = np.empty((neuron_nbr, neuron_nbr), dtype=list)
        adjacency_matrix = self.SOM.neural_adjacency_matrix
        for i in range(neuron_nbr):
            for j in range(neuron_nbr):
                positions[i, j] = project(self.SOM.nodes[i, j].weight[0], self.SOM.nodes[i, j].weight[1])
                draw_data_point(self.canvas, positions[i, j][0], positions[i, j][1], "blue")
                for k in range(i*neuron_nbr+j):
                    if adjacency_matrix[i*neuron_nbr+j][k] == 1:
                        self.canvas.create_line(positions[i, j][0], positions[i, j][1], positions[k//neuron_nbr, k%neuron_nbr][0], positions[k//neuron_nbr, k%neuron_nbr][1], fill="blue")

    def draw_metrics(self):
        self.ite_str = StringVar()
        self.epoch_str = StringVar()
        self.neurons_str = StringVar()
        self.dataset_str = StringVar()
        self.mean_error_str = StringVar()
        self.psnr_str = StringVar()

        self.ite_label = Label(textvariable=self.ite_str)
        self.epoch_label = Label(textvariable=self.epoch_str)
        self.neurons_label = Label(textvariable=self.neurons_str)
        self.dataset_label = Label(textvariable=self.dataset_str)
        self.mean_error_label = Label(textvariable=self.mean_error_str)
        self.psnr_label = Label(textvariable=self.psnr_str)

        self.ite_label.grid(column=10, row=0)
        self.epoch_label.grid(column=10, row=1)
        self.neurons_label.grid(column=10, row=2)
        self.dataset_label.grid(column=10, row=3)
        self.mean_error_label.grid(column=10, row=4)
        self.psnr_label.grid(column=10, row=5)
        self.update_metrics()

    def update_metrics(self):
        self.ite_str.set("Iteration number : "+str(self.current_iteration))
        self.epoch_str.set("Epochs number : "+str(self.current_iteration//self.epoch_time))
        self.neurons_str.set("Neurons number : "+str(neuron_nbr*neuron_nbr))
        self.dataset_str.set("Dataset size : "+str(dataset_size))
        winners = self.SOM.winners()
        self.mean_error_str.set("Mean error : "+str(self.SOM.compute_mean_error(winners)))
        self.psnr_str.set("PSNR : "+str(self.SOM.peak_signal_to_noise_ratio(winners)))

    def run_once(self):
        try:
            iterations = int(self.number.get())
        except ValueError:
            iterations = 0
        if self.listbox.get() == "epochs":
            iterations = iterations * self.epoch_time
        if iterations + self.current_iteration > self.total_iterations:
            iterations = self.total_iterations - self.current_iteration
        for i in range(iterations):
            if self.current_iteration % self.epoch_time == 0:
                self.SOM.generate_random_list()
            vect = self.SOM.unique_random_vector()
            self.SOM.train(self.current_iteration, self.epoch_time, vect)
            self.current_iteration += 1
        self.canvas.delete("all")
        self.draw_data()
        self.draw_map()
        self.canvas.update()
        self.update_metrics()

    def run(self):
        start_time = time.time()
        while self.current_iteration < self.total_iterations:
            if self.current_iteration % self.epoch_time == 0:
                self.SOM.generate_random_list()
                self.update_metrics()
            vect = self.SOM.unique_random_vector()
            self.SOM.train(self.current_iteration, self.epoch_time, vect)
            self.current_iteration += 1
            if time.time() - start_time > 0.1:
                self.canvas.delete("all")
                self.draw_data()
                self.draw_map()
                self.canvas.update()
                start_time = time.time()


GraphicalSOM()

from Model import *
import itertools
import multiprocessing as mp


def mutate(value, bounds):
    value += (np.random.rand() * 2 - 1) * mutation_value * (bounds[1] - bounds[0])
    if value > bounds[1]:
        value = bounds[1]
    elif value < bounds[0]:
        value = bounds[0]
    return value


class Genome:
    def __init__(self):
        self.epoch_nbr = np.random.random_integers(range_epoch_nbr[0], range_epoch_nbr[1])
        self.epsilon_start = (range_epsilon_start[1]-range_epsilon_start[0]) * np.random.random() + range_epsilon_start[0]
        self.epsilon_end = (range_epsilon_end[1]-range_epsilon_end[0]) * np.random.random() + range_epsilon_end[0]
        self.sigma_start = (range_sigma_start[1]-range_sigma_start[0]) * np.random.random() + range_sigma_start[0]
        self.sigma_end = (range_sigma_end[1]-range_sigma_end[0]) * np.random.random() + range_sigma_end[0]
        self.fitness = 255

    def crossover(self, father, mother):
        ratio = np.random.random()
        self.epoch_nbr = int(father.epoch_nbr * ratio + (1 - ratio) * mother.epoch_nbr)
        self.epsilon_start = father.epsilon_start * ratio + (1 - ratio) * mother.epsilon_start
        self.epsilon_end = father.epsilon_end * ratio + (1 - ratio) * mother.epsilon_end
        self.sigma_start = father.sigma_start * ratio + (1 - ratio) * mother.sigma_start
        self.sigma_end = father.sigma_end * ratio + (1 - ratio) * mother.sigma_end
        self.fitness = 255

    def mutation(self, source):
        self.epoch_nbr = source.epoch_nbr
        self.epsilon_start = source.epsilon_start
        self.epsilon_end = source.epsilon_end
        self.sigma_start = source.sigma_start
        self.sigma_end = source.sigma_end
        self.fitness = 255

        if probability_mutation > np.random.random():
            self.epoch_nbr = int(mutate(source.epoch_nbr, range_epoch_nbr))
        if probability_mutation > np.random.random():
            self.epsilon_start = mutate(source.epsilon_start, range_epsilon_start)
        if probability_mutation > np.random.random():
            self.epsilon_end = mutate(source.epsilon_end, range_epsilon_end)
        if probability_mutation > np.random.random():
            self.sigma_start = mutate(source.sigma_start, range_sigma_start)
        if probability_mutation > np.random.random():
            self.sigma_end = mutate(source.sigma_end, range_sigma_end)

    def run_fitness(self, data):
        print("Running "+str(self.epoch_nbr)+" epoch !")
        print(str(os.getpid()))
        epoch_time = len(data)
        nb_iter = epoch_time * self.epoch_nbr
        som = SOM(data, kohonen(), self.epsilon_start, self.epsilon_end, self.sigma_start, self.sigma_end, self.epoch_nbr)
        for i in range(nb_iter):
            if i % epoch_time == 0:
                som.generate_random_list()
            vector = som.unique_random_vector()
            som.train(i, epoch_time, vector)
        data_comp = som.winners()
        self.fitness = compute_mean_error(data_comp, data, som.get_som_as_list())

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness


class Population:
    def __init__(self):
        img = Dataset("./image/Audrey.png")
        self.data = img.data
        self.current = []
        for i in range(nb_individuals):
            self.current.append(Genome())

    def run(self):
        for i in range(nb_generations):
            print("Generation "+str(i)+"/"+str(nb_generations))
            self.evaluate_all()
            self.select()

    def evaluate_all(self):
        pool = mp.Pool(nb_individuals)
        pool.starmap(Genome.run_fitness, zip(self.current, itertools.repeat(self.data)))
        pool.close()
        pool.join()

    def select(self):
        self.current.sort()
        for j in self.current:
            print(j.fitness)
        best = self.current[0]
        print("Best error : "+str(best.fitness)+" in "+str(best.epoch_nbr)+" epochs")
        print("(eps_s: "+str(best.epsilon_start)+" eps_e"+str(best.epsilon_end)+" sig_s: "+str(best.sigma_start)+" sig_e"+str(best.sigma_end)+")")
        new = []
        for i in range(nb_individuals):
            if i <= elite_proportion*nb_individuals:
                new.append(self.current[i])
            else:
                child = Genome()
                child.crossover(self.current[np.random.randint(0, nb_individuals)], self.current[np.random.randint(0, nb_individuals)])
                child.mutation(child)
                new.append(child)
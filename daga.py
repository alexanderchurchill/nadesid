import random,pickle,copy,bisect,os,sys,pdb
import numpy as np
import theano
from denoising_autoencoder import dA
from custom_dataset import SequenceDataset
from optimizers import sgd_optimizer
import distance
from hiff import HIFF
from scipy.spatial.distance import pdist,cdist
from nade import NADE

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

class AESolver(object):
    """
    The Denoising Autoencoder Genetic Algorithm
    """
    def __init__(self,fitness_f):
        super(AESolver, self).__init__()
        self.FITNESS_F = fitness_f
        if self.FITNESS_F == "hiff":
            self.HIFF = HIFF(NUMGENES=128,K=2,P=7)
            self.fitness = self.hiff_fitness
        elif self.FITNESS_F == "knapsack":
            self.fitness = self.knapsack_fitness
        elif self.FITNESS_F == "max_ones":
            self.fitness = self.max_ones_fitness
        elif self.FITNESS_F == "left_ones":
            self.fitness = self.left_ones

    def generate_random_string(self,l=20):
        return [random.choice([0,1]) for i in range(l)]

    def knapsack_fitness(self,string):
        knapsack = self.knapsack
        weights = []
        for i,c in enumerate(knapsack.capacities):
            weights.append(np.sum(np.array(knapsack.constraints[i])*string))
        over = 0
        for i,w in enumerate(weights):
            if w > knapsack.capacities[i]:
                over += (w - knapsack.capacities[i])
        if over > 0:
            return -over
        else:
            _fitness = np.sum(np.array(knapsack.values)*string)
            return _fitness

    def hiff_fitness(self,string):
        fitness = self.HIFF.H(string)
        return fitness

    def max_ones_fitness(self,string):
        fitness = np.sum(string^self.mask)
        if cache:
            self.cache_fitness(fitness)
        return fitness

    def left_ones_fitness(self,_string):
        string =_string^self.mask
        fitness = sum(string[0:len(string)/2]) - sum(string[len(string)/2:])
        if cache:
            self.cache_fitness(fitness)
        return fitness

    def tournament_selection_replacement(self,
                                         population,
                                         fitnesses=None,
                                         pop_size=None):
        if pop_size == None:
            pop_size = len(population)
        if fitnesses == None:
            fitnesses = self.fitness_many(population)
        new_population = []
        while len(new_population) < pop_size:
            child_1 = int(np.random.random() * pop_size)
            child_2 = int(np.random.random() * pop_size)
            if fitnesses[child_1] > fitnesses[child_2]:
                new_population.append(copy.deepcopy(population[child_1]))
            else:
                new_population.append(copy.deepcopy(population[child_2]))
        return new_population

    def get_good_strings(self,strings,lim=20,unique=False,fitnesses=None):
        if fitnesses == None:
            fitnesses = [self.fitness(s) for s in strings]
        sorted_fitnesses = sorted(range(len(fitnesses)),
                                  key=lambda k: fitnesses[k])
        sorted_fitnesses.reverse()
        if unique == False:
            return ([strings[i] for i in sorted_fitnesses[0:lim]],
                    [fitnesses[k] for k in sorted_fitnesses[0:lim]])
        else:
            uniques = {}
            good_pop = []
            good_pop_fitnesses = []
            index = 0
            while len(good_pop) < lim and index < len(sorted_fitnesses):
                key = str(strings[sorted_fitnesses[index]])
                if key not in uniques:
                    uniques[key] = 0
                    good_pop.append(strings[sorted_fitnesses[index]])
                    good_pop_fitnesses.append(
                        fitnesses[sorted_fitnesses[index]]
                        )
                index += 1
            if len(good_pop) == lim:
                return [good_pop,good_pop_fitnesses]
            else:
                while len(good_pop) < lim:
                    good_pop.append(self.generate_random_string(
                                        l=len(strings[0]))
                                    )
                    good_pop_fitnesses.append(self.fitness(good_pop[-1]))
                return [good_pop,good_pop_fitnesses]

    def RTR(self,
            population,
            sampled_population,
            population_fitnesses,
            sample_fitnesses,
            w=None):
        if w == None:
            w = len(population)/20
        _population = np.array(population)
        for ind_i,individual in enumerate(sampled_population):
            indexes = np.random.choice(len(_population), w, replace=False)
            distances = cdist(_population[indexes],[individual],"hamming")
            replacement = indexes[np.argmin(distances.flatten())]
            if population_fitnesses[replacement] < sample_fitnesses[ind_i]:
                _population[replacement] = individual
                population_fitnesses[replacement] = sample_fitnesses[ind_i]
        return _population

    def fitness_many(self,strings):
        return [self.fitness(s) for s in strings]

    def train_dA(self,
                 data,
                 corruption_level=0.2,
                 num_epochs=200,
                 lr=0.1,
                 output_folder="",
                 iteration=0):
        train_data = data
        train_set = SequenceDataset(train_data,batch_size=20,number_batches=None)
        sgd_optimizer(self.dA.params,[self.dA.input],self.dA.cost,train_set,
                      lr=lr,num_epochs=num_epochs,save=False,
                      output_folder=output_folder,iteration=iteration)

    def train_NADE(self,
                 data,
                 num_epochs=200,
                 lr=0.1,
                 output_folder="",
                 iteration=0):
        train_data = data
        train_set = SequenceDataset(train_data,batch_size=20,number_batches=None)
        sgd_optimizer(self.NADE.params,[self.NADE.v],self.NADE.cost,train_set,
                      lr=lr,num_epochs=num_epochs,save=False,
                      output_folder=output_folder,iteration=iteration)

    def build_sample_dA(self):  
        self.sample_dA = theano.function([self.dA.input],self.dA.sample)

    def iterative_algorithm(
        self,
        name,
        pop_size=100,
        genome_length=20,
        lim_percentage=20,
        corruption_level=0.2,
        num_epochs=50,
        lr = 0.1,
        max_evaluations=200000,
        unique_training=False,
        hiddens=300,
        rtr = True,
        w=10
        ):
        results_path = "results/autoencoder/{0}/".format(name)
        ensure_dir(results_path)
        fitfile = open("{0}fitnesses.dat".format(results_path),"w")
        self.mask = np.random.binomial(1,0.5,genome_length)
        trials = max_evaluations/pop_size
        population_limit = int(pop_size*(lim_percentage/100.0))
        # self.dA = dA(n_visible=genome_length,n_hidden=hiddens)
        # self.dA.build_dA(corruption_level)
        # self.build_sample_dA()
        self.NADE = NADE(n_visible=genome_length,n_hidden=hiddens)
        # self.NADE.build_NADE()
        new_population = np.random.binomial(1,0.5,(pop_size,genome_length))
        self.population_fitnesses = self.fitness_many(new_population)
        fitfile.write("{0},{1},{2},{3}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses),np.std(self.population_fitnesses)))
        for iteration in range(0,trials):
            print "iteration:",iteration
            population = new_population
            self.population = new_population
            rw = self.tournament_selection_replacement(population)
            good_strings,good_strings_fitnesses=self.get_good_strings(
                                          population,
                                          population_limit,
                                          unique=unique_training,
                                          fitnesses=self.population_fitnesses
                                        )
            print "training A/E"
            training_data = np.array(good_strings)
            self.train_NADE(training_data,
                          num_epochs=num_epochs,
                          lr=lr)
            print "sampling..."
            sampled_population = np.array(self.NADE.sample_multiple(n=len(new_population)),"b")
            self.sample_fitnesses = self.fitness_many(sampled_population)
            if rtr:
                new_population = self.RTR(
                              population,
                              sampled_population,
                              population_fitnesses=self.population_fitnesses,
                              sample_fitnesses=self.sample_fitnesses,
                              w=w
                              )
            else:
                new_population = sampled_population
                new_population[0:1] = good_strings[0:1]
                self.population_fitnesses = self.sample_fitnesses
                self.population_fitnesses[0:1] = good_strings_fitnesses[0:1]
            print "{0},{1},{2}\n".format(np.mean(self.population_fitnesses),
                                         np.min(self.population_fitnesses),
                                         np.max(self.population_fitnesses))
            print "best from previous:",(
              self.fitness(new_population[np.argmax(self.population_fitnesses)])
                )
            fitfile.write("{0},{1},{2},{3}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses),np.std(self.population_fitnesses)))
            fitfile.flush()
        fitfile.close()
        return new_population

if __name__ == '__main__':
    ae = AESolver("hiff")
    args = sys.argv
    pop_size = int(args[1])
    lim_percentage = int(args[2])
    num_epochs = int(args[3])
    lr = float(args[4])
    hiddens = int(args[5])
    rtr = int(args[6])
    if rtr == 1:
        rtr = True
    else:
        rtr = False
    w = int(args[7])
    trial = int(args[8])
    name = "hiff-128-{0}".format("-".join([str(s) for s in [pop_size,
                                                            lim_percentage,
                                                            num_epochs,
                                                            lr,
                                                            hiddens,
                                                            rtr,
                                                            w,
                                                            trial]
                                                            ]))
    ae.iterative_algorithm(
        name,
        pop_size=pop_size,
        genome_length=128,
        lim_percentage=lim_percentage,
        corruption_level=0.05,
        num_epochs=num_epochs,
        lr = lr,
        max_evaluations=200000,
        unique_training=True,
        hiddens=hiddens,
        rtr = rtr,
        w=w
        )

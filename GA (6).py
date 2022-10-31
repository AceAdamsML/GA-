
from random import randint
from copy import deepcopy
import random
import matplotlib.pyplot as plt
import numpy as np


class Individual:
    def __init__(self, genome_size, genome_min, genome_max):
        self.genome_min = genome_min
        self.genome_max = genome_max
        self.genome_size = genome_size
        self.genome = []
        self.fitness = 0
        self.probability = 0
        for i in range(genome_size):
            self.genome.append(random.random())

    def get_probability(self):
        return self.probability

    def set_probability(self, p):
        self.probability = p

    # def calculate_fitness(self):
    #     self.fitness = sum(self.genome)

    def calculate_fitness_WS4_1(self):
        SUM = 0
        for i in range(0, self.genome_size - 1):
            SUM += 100 * (self.genome[i + 1] - self.genome[i] ** 2) ** 2 + (1 - self.genome[i]) ** 2
        self.fitness = SUM

    def calculate_fitness_WS4_2(self):
        TERM1_SUM = 0
        TERM2_SUM = 0
        for i in range(0, self.genome_size):
            TERM1_SUM += self.genome[i] ** 2
            TERM2_SUM += np.cos(2 * np.pi * self.genome[i])
        TERM1_SUM = -20 * np.exp(-0.2 * np.sqrt(TERM1_SUM / self.genome_size))
        TERM2_SUM = np.exp(TERM2_SUM / self.genome_size)

        self.fitness = TERM1_SUM - TERM2_SUM

    def get_fitness(self):
        return self.fitness

    def mutate(self, chance, mut_step):
        i = 0
        for gene in self.genome:
            if random.random() < chance:
                alter = random.random() * mut_step
                if randint(1, 2) % 2:
                    self.genome[i] += alter
                else:
                    self.genome[i] -= alter

                if self.genome[i] > self.genome_max:
                    self.genome[i] = self.genome_max
                if self.genome[i] < self.genome_min:
                    self.genome[i] = self.genome_min

            i += 1

    def mutate_guas(self, chance, mut_step):
        i = 0
        for gene in self.genome:
            if random.random() < chance:
                alter = random.gauss(0, mut_step)

                self.genome[i] *= alter

                if self.genome[i] > self.genome_max:
                    self.genome[i] = self.genome_max
                if self.genome[i] < self.genome_min:
                    self.genome[i] = self.genome_min

            i += 1


class Population:
    def __init__(self, pop_size, genome_size, genome_min, genome_max):
        self.individuals = []
        self.pop_size = pop_size
        self.genome_size = genome_size
        for i in range(pop_size):
            self.individuals.append(Individual(genome_size, genome_min, genome_max))

    def evaluate(self, fitness_funtion_type):
        for ind in self.individuals:
            if fitness_funtion_type == 1:
                ind.calculate_fitness_WS4_1()
            if fitness_funtion_type == 2:
                ind.calculate_fitness_WS4_2()

    def cross_over(self):
        children = []
        for i in range(0, self.pop_size - 1, 2):
            parent1 = deepcopy(self.individuals[i])
            parent2 = deepcopy(self.individuals[i + 1])
            splice_idx = randint(1, self.genome_size - 2)
            temp = deepcopy(parent1.genome[splice_idx:])

            parent1.genome[splice_idx:] = deepcopy(parent2.genome[splice_idx:])
            parent2.genome[splice_idx:] = deepcopy(temp)

            children.append(parent1)
            children.append(parent2)
        self.individuals = deepcopy(children)

    def cross_over_multipoint(self):
        children = []
        for i in range(0, self.pop_size - 1, 2):
            parent1 = deepcopy(self.individuals[i])
            parent2 = deepcopy(self.individuals[i + 1])
            splice_idx = randint(1, self.genome_size // 2)
            temp = deepcopy(parent1.genome[splice_idx:])

            parent1.genome[splice_idx:] = deepcopy(parent2.genome[splice_idx:])
            parent2.genome[splice_idx:] = deepcopy(temp)

            splice_idx = randint(splice_idx + 1, self.genome_size - 2)
            temp = deepcopy(parent1.genome[splice_idx:])

            parent1.genome[splice_idx:] = deepcopy(parent2.genome[splice_idx:])
            parent2.genome[splice_idx:] = deepcopy(temp)

            children.append(parent1)
            children.append(parent2)
        self.individuals = deepcopy(children)

    def cross_over_arithmetic(self):
        a = 0.6
        children = []

        for i in range(0, self.pop_size - 1, 2):
            parent1 = deepcopy(self.individuals[i])  # Get every 2 individuals as parents
            parent2 = deepcopy(self.individuals[i + 1])

            child1 = deepcopy(parent1)
            child2 = deepcopy(parent2)

            for n in range(0, self.genome_size):
                child1.genome[n] = a * parent1.genome[n] + (1 - a) * parent2.genome[n]
                child2.genome[n] = a * parent2.genome[n] + (1 - a) * parent1.genome[n]

            children.append(child1)
            children.append(child2)

        self.individuals = deepcopy(children)

    def tournament(self):
        children = []
        for i in range(self.pop_size):
            ind1 = self.individuals[randint(0, self.pop_size - 1)]
            ind2 = self.individuals[randint(0, self.pop_size - 1)]
            if ind1.get_fitness() < ind2.get_fitness():
                children.append(deepcopy(ind1))
            else:
                children.append(deepcopy(ind2))

        self.individuals = deepcopy(children)

    def roulette_wheel(self):
        children = []
        min_fit = min([i.get_fitness() for i in self.individuals])
        new_fit = [1 + abs(min_fit) + i.get_fitness() for i in self.individuals]
        sum_fitness = sum([1 / fit for fit in new_fit])

        for i in range(self.pop_size):
            j = 0

            running_fit = 0
            selection_pt = random.random() * sum_fitness

            while running_fit < selection_pt:
                running_fit += 1 / new_fit[j]

                j += 1

            children.append(self.individuals[j - 1])

        self.individuals = deepcopy(children)

    def mutate(self, chance, mut_step):
        for i in self.individuals:
            # i.mutate(chance, mut_step)
            i.mutate_guas(chance, mut_step)


    def get_highest_fitness(self):
        fitness = 0
        for i in self.individuals:
            if i.get_fitness() > fitness:
                fitness = i.get_fitness()
        return fitness

    def get_lowest_fitness(self):
        fitness = 10000
        for i in self.individuals:
            if i.get_fitness() < fitness:
                fitness = i.get_fitness()
        return fitness

    def get_avg_fitness(self):
        total_fitness = 0
        for i in self.individuals:
            total_fitness += i.get_fitness()
        return total_fitness / self.pop_size
    def gen(self):
      pass


def run(total_gen,genome_size,pop_size,mutation_rate,mut_step):
  high_fit_list = []

  avg_fit_list = []
  population = Population(pop_size, genome_size, genome_min, genome_max)
  for g in range(total_gen):
      population.evaluate(fitness_funtion_type)
      highest_fit = population.get_lowest_fitness()
      high_fit_list.append(highest_fit)
      avg_fit = population.get_avg_fitness()
      avg_fit_list.append(avg_fit)
      # population.tournament()
      population.roulette_wheel()
      population.cross_over_arithmetic()
      population.mutate(mutation_rate, mut_step)
  
      # print("\nGeneramrion:", g)
      # print("Highest Fitness:", highest_fit)
      # print("Population Average Fitness:", avg_fit)
      
  
  return high_fit_list
def run1(mut_steps):
  for mut_step in mut_steps:

    high_fit_list = []


    plt.title("Using Optimization Funtion: " + str(fitness_funtion_type))
    plt.xlabel('Generation')
    plt.ylabel('Fitness')

    high_fit_list=run(total_gen,genome_size,pop_size,mutation_rate,mut_step)
    plt.plot(high_fit_list, label="Mutation rate " + str(mutation_rate) + " Mutation step "+ str(mut_step) )

  plt.legend()
  plt.show()

def run2(pop_sizes):
  for pop_size in pop_sizes:
    mut_step=1
    high_fit_list = []


    plt.title("Using Optimization Funtion: " + str(fitness_funtion_type))
    plt.xlabel('Generation')
    plt.ylabel('Fitness')

    high_fit_list=run(total_gen,genome_size,pop_size,mutation_rate,mut_step)
    plt.plot(high_fit_list, label="Mutation rate " + str(mutation_rate) + " Population Size "+ str(pop_size))

  plt.legend()
  plt.show()

def run3(mutation_rates):
  for mutation_rate in mutation_rates:
    mut_step=1
    high_fit_list = []


    plt.title("Using Optimization Funtion: " + str(fitness_funtion_type))
    plt.xlabel('Generation')
    plt.ylabel('Fitness')

    high_fit_list=run(total_gen,genome_size,pop_size,mutation_rate,mut_step)
    plt.plot(high_fit_list, label="Mutation rate " + str(mutation_rate) + " Population Size "+ str(pop_size))

  plt.legend()
  plt.show()

def run4(total_gens_):
  for total_gens in total_gens_:
    mut_step=1
    high_fit_list = []


    plt.title("Using Optimization Funtion: " + str(fitness_funtion_type))
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    
    high_fit_list=run(total_gen,genome_size,pop_size,mutation_rate,mut_step)
    plt.plot(high_fit_list, label="Mutation rate " + str(mutation_rate) + " Total Gens "+ str(total_gens))

  plt.legend()
  plt.show()




fitness_funtion_type = 1

if fitness_funtion_type == 1:
    genome_min = -100
    genome_max = 100
if fitness_funtion_type == 2:
    genome_min = -32
    genome_max = 32


total_gen = 1000
genome_size = 20
pop_size = 50
mutation_rate = 0.05

mut_step = 1

high_fit_list = []

avg_fit_list = []


population = Population(pop_size, genome_size, genome_min, genome_max)
for g in range(total_gen):
    population.evaluate(fitness_funtion_type)
    highest_fit = population.get_lowest_fitness()
    high_fit_list.append(highest_fit)
    avg_fit = population.get_avg_fitness()
    avg_fit_list.append(avg_fit)
    # population.tournament()
    population.roulette_wheel()
    population.cross_over_arithmetic()
    population.mutate(mutation_rate, mut_step)

    print("\nGeneramrion:", g)
    print("Highest Fitness:", highest_fit)
    print("Population Average Fitness:", avg_fit)

plt.title("Using Optimization Funtion: " + str(fitness_funtion_type))
plt.plot(avg_fit_list, label="Average Fitness")
plt.plot(high_fit_list, label="Minimum Fitness")
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.show()





total_gen = 200
genome_size = 20
pop_size = 50
mutation_rate = 0.1

mut_steps=[0.1,0.2,0.3,0.5,1,3,5]
pop_sizes=[10,20,30,50,100]
mutation_rates=[0.1,0.2,0.3,0.5,1]
total_gens=[10,50,100,150,200]

run1(mut_steps) 
run2(pop_sizes) 
run3(mutation_rates)
run4(total_gens)



high_fit_list = []

avg_fit_list = []
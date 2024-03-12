import random

import torch

############################################################
#
#  Prepare constants
#
############################################################

print("\nSetting up environment...")
# specifies the problem to solve from <1;10>
problem = "1"
# specifies the population size to use
POPULATION_SIZE = 2
# population must be divisible by 2, fix it just in case
POPULATION_SIZE += POPULATION_SIZE % 2
# number of generations
MAX_GENERATIONS = 1000
# initial number of buses to use in initial population
starting_buses = 10
# amount of individuals from a population to pick and transfer to new population
best_percent_amount = int(POPULATION_SIZE * 0.1)
# 50% of individuals from a population
percent_to_mate_amount = int(POPULATION_SIZE * 0.5)
# set seed to use for generating random numbers
seed = torch.initial_seed()
torch.manual_seed(seed)
# specifies the device to use, GPU: "cuda", CPU: "cpu"
device = torch.device("cpu")

############################################################
#
#  Load Data
#
############################################################

csvSpojeFile = open("input/spoje_id_DS{}_J_1.csv".format(problem), "r")
csvSpojeFile.readline()
csvSpojeFileLines = csvSpojeFile.readlines()
csvDistMatrixFile = open("input/Tij_DS{}_J_1.csv".format(problem), "r")
n = int(csvDistMatrixFile.readline())
m = int(csvDistMatrixFile.readline())

start_vec = torch.tensor([int(line.split(";")[6]) for line in csvSpojeFileLines if len(line) > 0], device=device)
end_vec = torch.tensor([int(line.split(";")[7]) for line in csvSpojeFileLines if len(line) > 0], device=device)
dist_matrix = torch.tensor([[int(value) for value in csvDistMatrixFile.readline().split(";") if value != "\n"] for j in range(n)], device=device)

# used to select spoje in a solution
spoje_indices_vec = torch.ones(m, device=device)
spoje_indices_vec[0] = 0
spoje_indices_vec = spoje_indices_vec.cumsum(0).int()

############################################################
#
#  Create initial population
#
############################################################

population_arr = [[0 for j in range(m)] for i in range(POPULATION_SIZE)]

for i in range(POPULATION_SIZE):
    buses_last_spoj = [-1] * starting_buses

    for j in range(m):
        is_valid = False
        rand_bus = random.randint(0, starting_buses - 1)

        while not is_valid:
            rand_bus = random.randint(0, starting_buses - 1)
            is_valid = buses_last_spoj[rand_bus] == -1 or end_vec[buses_last_spoj[rand_bus]] + dist_matrix[buses_last_spoj[rand_bus], j] <= start_vec[j]

        population_arr[i][j] = rand_bus
        buses_last_spoj[rand_bus] = j

# represents population, one row is one solution, one column is one spoj
population = torch.tensor(population_arr, device=device)
# a mirror of population representing each solution as a matrix mask with 1 where bus index and spoj index meet
population_matrix = torch.zeros(POPULATION_SIZE, n, m, device=device)

print(population)

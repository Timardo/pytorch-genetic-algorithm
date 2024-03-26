import random

import torch

############################################################
#
#  Prepare constants
#
############################################################

print("\nSetting up environment...")
# specifies the problem to solve from <0;10>, 0 is a dummy problem meant for testing with readable output, others are real data
problem = "1"
# specifies the population size to use
POPULATION_SIZE = 20
# population must be divisible by 2, fix it just in case
POPULATION_SIZE += POPULATION_SIZE % 2
# number of generations
MAX_GENERATIONS = 1000
# initial number of buses to use in initial population
STARTING_BUSES_COUNT = 100
# amount of individuals from a population to pick and transfer to new population
best_percent_amount = int(POPULATION_SIZE * 0.1)
# 50% of individuals from a population
percent_to_mate_amount = int(POPULATION_SIZE * 0.5)
# set seed to use for generating random numbers
seed = torch.initial_seed()
torch.manual_seed(seed)
# specifies the device to use, GPU: "cuda", CPU: "cpu"
device = torch.device("cuda")

############################################################
#
#  Load Data
#
############################################################

csv_bus_lines_file = open("input/spoje_id_DS{}_J_1.csv".format(problem), "r")
# ignore csv header line
csv_bus_lines_file.readline()
csvBusLinesFileLines = csv_bus_lines_file.readlines()
csvDistMatrixFile = open("input/Tij_DS{}_J_1.csv".format(problem), "r")
# load the number of bus lines this matrix consists of
bus_line_count = int(csvDistMatrixFile.readline())
# this should be ALWAYS the same as bus_line_count, ignore
csvDistMatrixFile.readline()


start_times_vec = torch.tensor([int(line.split(";")[6]) for line in csvBusLinesFileLines if len(line) > 0], device=device)
end_times_vec = torch.tensor([int(line.split(";")[7]) for line in csvBusLinesFileLines if len(line) > 0], device=device)
dist_matrix = torch.tensor([[int(value) for value in csvDistMatrixFile.readline().split(";") if value != "\n"] for j in range(bus_line_count)], device=device)
transition_matrix = end_times_vec.unsqueeze(-1).expand(bus_line_count, bus_line_count).add(dist_matrix).lt(start_times_vec)

# serve as a static index matrices for creating list of solution matrices from a population in order to get the fitness of solutions (number of buses)
population_indices = torch.arange(POPULATION_SIZE, device=device).unsqueeze(-1).expand(POPULATION_SIZE, bus_line_count)
row_indices = torch.arange(bus_line_count, device=device)
# a mirror of population representing each solution as a matrix mask with 1 where bus index and bus line index meet
population_matrix = torch.zeros(POPULATION_SIZE, bus_line_count, STARTING_BUSES_COUNT, device=device, dtype=torch.int8)
# chance to mutate a single element (change the bus that is serving a specific bus lane to a random bus that already has some bus lanes)
# base chance to mutate a solution is 80%, applied globally must be divided by number of bus lanes
# THIS DOES NOT GUARANTEE A MAXIMUM OF ONE MUTATION PER SOLUTION!
mutation_chance = 0.8 / bus_line_count

############################################################
#
#  Create initial population
#
############################################################

population_arr = [[0 for j in range(bus_line_count)] for i in range(POPULATION_SIZE)]

for i in range(POPULATION_SIZE):
    buses_last_bus_lane = [-1] * STARTING_BUSES_COUNT

    for j in range(bus_line_count):
        is_valid = False
        rand_bus = random.randint(0, STARTING_BUSES_COUNT - 1)

        while not is_valid:
            rand_bus = random.randint(0, STARTING_BUSES_COUNT - 1)
            is_valid = True#buses_last_bus_lane[rand_bus] == -1 or end_times_vec[buses_last_bus_lane[rand_bus]] + dist_matrix[buses_last_bus_lane[rand_bus], j] <= start_times_vec[j]

        population_arr[i][j] = rand_bus
        buses_last_bus_lane[rand_bus] = j

# represents population, one row is one solution, one column is one bus lane with the number indicating which bus is serving this lane
population = torch.tensor(population_arr, device=device)

############################################################
#
#  Calculate fitness function
#  - this version of the fitness function does not take into account the level of inadmissibility of solutions of population and expect them to be admissible
#
############################################################

population_matrix.mul_(0)
# set 1 to indices that indicate which buses serve which bus line
population_matrix[population_indices, row_indices, population] = 1
# get fitness tensor (number of buses) by summing the number of server bus lines for each bus, then summing the number of non-zero-bus-line-serving buses and extracting this number
fitness = population_matrix.cumsum_(1).gt_(0).cumsum_(2)[:, bus_line_count - 1, STARTING_BUSES_COUNT - 1]

############################################################
#
#  Sort population
#
############################################################

# sort population according to the fitness function
population = population[fitness.sort(0, descending=False)[1]]

############################################################
#
#  Crossover operator
#  - this version of the crossover operator produces non-admissible solutions that must be fixed or their level of inadmissibility must be taken into account in the fitness function
#
############################################################

# extract best part of the population to separate tensor
population_best_ten_percent = population[:best_percent_amount]
# initialize random indices tensor and shuffle the rows of the population for more random crossover
population_shuffle_indices = torch.randperm(POPULATION_SIZE, device=device)
# duplicate best half of the population over the worst half
population[percent_to_mate_amount:] = population[:percent_to_mate_amount]
# clone population to create a pair for each solution
population_2 = torch.clone(population)
# shuffle both populations
population = population[population_shuffle_indices, :]
population_shuffle_indices = torch.randperm(POPULATION_SIZE, device=device)
population_2 = population_2[population_shuffle_indices, :]
# create mask for picking between populations
population_crossover_mask = torch.rand(POPULATION_SIZE, bus_line_count, device=device).lt_(0.5).bool()
# apply mask to pick items from the first population
population.mul_(population_crossover_mask)
# negate the mask
population_crossover_mask.bitwise_not_()
# apply again on the other population
population_2.mul_(population_crossover_mask)
# merge together
population.add_(population_2)
# transfer the best part of the population back
population[:best_percent_amount] = population_best_ten_percent

############################################################
#
#  Mutation operator
#  - this version of the mutation operator produces non-admissible solutions that must be fixed or their level of inadmissibility must be taken into account in the fitness function
#
############################################################

# prepare the mask
population_mutation_mask = torch.rand(POPULATION_SIZE, bus_line_count, device=device).lt_(mutation_chance).bool()
# clear population matrix
population_matrix.mul_(0)
# set 1 to indices that indicate which buses serve which bus line
population_matrix[population_indices, row_indices, population] = 1
# sort population matrix in a way to get the count of buses for each solution as well as the possible bus indices for each solution
# each possible_indices[i] vector's first X values where X is represented by buses_counts[i] represent the possible bus indices
# that the solution can mutate to without changing the fitness (not mutating to empty buses)
buses_counts, possible_indices = population_matrix.sort(1, descending=True)[0][:, 0].sort(1, descending=True)
# create index matrix to use for multiplying random float values to get random indices for bus selection
max_bus_indices = buses_counts.cumsum_(1)[:, STARTING_BUSES_COUNT - 1].unsqueeze(-1).expand(POPULATION_SIZE, bus_line_count)
# create indices to use for mapping to possible buses
population_mutation_indices = torch.rand(POPULATION_SIZE, bus_line_count, device=device).mul_(max_bus_indices).int()
# finally represents a new completely random population filtered with the population_mutation_mask
possible_indices = possible_indices[population_indices, population_mutation_indices].mul_(population_mutation_mask)
# flip the mask
population_mutation_mask.bitwise_not_()
# apply the mask on population and combine with the mutation population
population.mul_(population_mutation_mask).add_(possible_indices)

############################################################
#
#  Calculate fitness function
#  - this version of the fitness function takes into account the level of inadmissibility of solutions of population
#  - each incorrect (impossible) transition between two bus lines counts as +1 to fitness function which means one more bus would be needed for a valid solution
#  - fixing an inadmissible solution is just as complex as using a mutation operator that produces admissible solutions only
#  - this is just a proof of concept that is not usable on large problems with large populations due to insanely high memory requirements
#  - as an example, the largest problem with around 1000 bus lanes and 100 starting buses can be run with maximum POPULATION_SIZE of 26 on a system with 64GB RAM and 12GB GPU VRAM
#    using ~42GB of shared GPU memory out of 43GB available (using more memory than the GPU can provide dramatically slows down the process anyway and should be avoided at all costs)
#
############################################################

population_matrix.mul_(0)
# set 1 to indices that indicate which buses serve which bus line
population_matrix[population_indices, row_indices, population] = 1
# indices matrix used to create 4D matrix that defines current transitions between bus lines
population_indices_1 = torch.arange(POPULATION_SIZE, device=device).unsqueeze(-1).unsqueeze(-1).expand(POPULATION_SIZE, STARTING_BUSES_COUNT, bus_line_count)
# indices matrix used to create 4D matrix that defines current transitions between bus lines
row_indices_1 = torch.arange(STARTING_BUSES_COUNT, device=device).unsqueeze(-1).expand(STARTING_BUSES_COUNT, bus_line_count)
# transposed population matrix for ease of indexing
population_matrix_transposed = population_matrix.transpose(1, 2)
# final mask containing all current bus line transitions filtered to upper part of diagonal (diagonal-exclusive)
# the size of this mask is POPULATION_SIZE * STARTING_BUSES_COUNT * bus_line_count * bus_line_count, the largest problem is 1000 bus lines and a minimum of ~60 buses
# this makes this mask the size of 60_000_000 * POPULATION_SIZE bytes which is about 60MB per individual in population which is not viable for large population sizes
# TODO: find an alternative that does not take as much space
population_matrix_transposed_mask = population_matrix_transposed[population_indices_1, row_indices_1].bitwise_and(population_matrix_transposed[population_indices_1, row_indices_1].transpose(2, 3)).triu(diagonal=1)
# current number of transitions in population
population_current_transitions = population_matrix_transposed_mask.cumsum(2).cumsum(3)[:, :, bus_line_count - 1, bus_line_count - 1].cumsum(1)[:, STARTING_BUSES_COUNT - 1]
# current transitions filtered with possible transitions mask omitting impossible transitions
population_possible_transitions = population_matrix_transposed_mask.bitwise_and(transition_matrix).cumsum(2).cumsum(3)[:, :, bus_line_count - 1, bus_line_count - 1].cumsum(1)[:, STARTING_BUSES_COUNT - 1]
# the difference between the number of current transitions and possible transitions, anything above zero is a non-admissible solution
population_transition_difference = population_current_transitions.add_(population_possible_transitions.mul_(-1))
# get fitness tensor (number of buses) by summing the number of server bus lines for each bus, then summing the number of non-zero-bus-line-serving buses and extracting this number
fitness = population_matrix.cumsum_(1).gt_(0).cumsum_(2)[:, bus_line_count - 1, STARTING_BUSES_COUNT - 1]
# add the difference to fitness
fitness.add_(population_transition_difference)

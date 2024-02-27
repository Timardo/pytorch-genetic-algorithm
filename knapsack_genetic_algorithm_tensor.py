import torch
from timeit import default_timer as timer
import math
import random

############################################################
#
#  Prepare constants
#
############################################################

# specifies the device to use, GPU: "cuda", CPU: "cpu"
device = torch.device("cuda")
# specifies the population size to use
POPULATION_SIZE = 4096
# population must be divisible by 2, fix it just in case
POPULATION_SIZE += POPULATION_SIZE % 2
# specifies the knapsack size in number of items
KNAPSACK_SIZE = 4096
# number of generations
max_generations = 4096
# amount of individuals from a population to pick and transfer to new population
best_percent_amount = int(KNAPSACK_SIZE * 0.1)
# 50% of individuals from a population
percent_to_mate_amount = int(POPULATION_SIZE * 0.5)
# chance to mutate a single item (flip its value), base is 80% per individual in the population, applied globally must be divided by knapsack size
mutation_chance = 0.8 / KNAPSACK_SIZE
# random values between 0 and 1 below this threshold will be interpreted as an item in the knapsack creating initial population
initial_population_random_threshold = torch.tensor(1 / 8, device=device)
# minimum weight of an item in a knapsack to generate
min_item_weight = int(math.log10(KNAPSACK_SIZE) * math.pow(KNAPSACK_SIZE, 1 / 4))
# maximum weight of an item in a knapsack to generate
max_item_weight = int(math.log10(KNAPSACK_SIZE) * math.pow(KNAPSACK_SIZE, 1 / 2))
# minimum price of an item in a knapsack to generate
min_item_price = 2
# maximum price of an item in a knapsack to generate
max_item_price = 100
# maximum weight of all items in a knapsack
max_knapsack_weight = round((min_item_weight + max_item_weight) / (random.uniform(3, 5)) * KNAPSACK_SIZE)

############################################################
#
#  Prepare knapsack data
#
############################################################

knapsack_prices = torch.randint(min_item_price, max_item_price + 1, (KNAPSACK_SIZE, ), device=device)
knapsack_weights = torch.randint(min_item_weight, max_item_weight + 1, (KNAPSACK_SIZE, ), device=device)
knapsack_max_weight = torch.tensor([max_knapsack_weight], device=device)
print("\nMax weight:")
print(knapsack_max_weight)

############################################################
#
#  Create target solution
#
############################################################

knapsack_coefficients_sorted_indices = torch.div(knapsack_prices, knapsack_weights).sort(descending=True)[1]
knapsack_prices_sorted = knapsack_prices[knapsack_coefficients_sorted_indices]
knapsack_weights_sorted = knapsack_weights[knapsack_coefficients_sorted_indices]
knapsack_weights_sorted_cumsum = torch.cumsum(knapsack_weights_sorted, 0)
target_solution = torch.max(torch.le(knapsack_weights_sorted_cumsum, max_knapsack_weight).int().mul_(torch.cumsum(knapsack_prices_sorted, 0)))
target_ceil = torch.max(knapsack_weights_sorted_cumsum.add_(-knapsack_weights_sorted).mul(-1).add_(max_knapsack_weight).float().div_(knapsack_weights_sorted.float()).clamp(0, 1).mul_(knapsack_prices_sorted).cumsum_(0)).float()

############################################################
#
#  Prepare initial population
#
############################################################

population_rand = torch.rand(POPULATION_SIZE, KNAPSACK_SIZE, device=device)
population = torch.lt(population_rand, initial_population_random_threshold).to(dtype=torch.int8, device=device)

############################################################
#
#  Extract best solution (most probably is admissible)
#  - extracts and prints current best solution
#
############################################################

print("\nCurrent best solution:")
print(torch.max(torch.mul(population, knapsack_prices).cumsum_(1)[:, KNAPSACK_SIZE - 1]))
print("Timer starts...")
start = timer()

for x in range(max_generations):
    ############################################################
    #
    #  Correct non-admissible solutions
    #  - first shuffles all items in all tensors accordingly using a random index permutation tensor to avoid bias - items with higher index have higher chance to be dropped
    #  - calculates cumulative sum of weights and creates a mask to apply on population to remove excessive items
    #
    ############################################################

    random_indices_knapsack = torch.randperm(KNAPSACK_SIZE, device=device)
    # shuffle items
    knapsack_weights = knapsack_weights[random_indices_knapsack]
    knapsack_prices = knapsack_prices[random_indices_knapsack]
    population = population[:, random_indices_knapsack]

    item_mask = torch.mul(population, knapsack_weights).cumsum_(1).le_(max_knapsack_weight)
    population.bitwise_and_(item_mask)

    ############################################################
    #
    #  Sort population
    #  - sorts whole population based on the cumulative sum of individual's item's prices which is at the last index of each individual
    #
    ############################################################

    # output   = input     [mul operation to get prices          ) cumulate prices[select last column] sort by rows desc.  )[get indices tensor] select by indices' tensor]
    population = population[torch.mul(population, knapsack_prices).cumsum_(1)[:, KNAPSACK_SIZE - 1].sort(0, descending=True)[1]]

    ############################################################
    #
    #  Crossover operation
    #  - after sorting, save best 10%, duplicate best 50% over worst 50%, shuffle rows, duplicate whole tensor, shuffle rows again
    #  - create true/false mask, apply bitwise_and on both populations and merge with bitwise_or
    #  - next apply mutation through true/false mask with bitwise_xor to flip 0 to 1 and 1 to 0 if the mask has 1
    #
    ############################################################

    # extract best part of the population to separate tensor
    population_best_ten_percent = population[:best_percent_amount]

    # duplicate best half of the population over worst half
    population[percent_to_mate_amount:] = population[:percent_to_mate_amount]
    # initialize random indices tensor and shuffle the rows of the population for more uniform and random crossover
    random_indices_population = torch.randperm(POPULATION_SIZE, device=device)
    population = population[random_indices_population, :]
    # copy tensor and do the same
    population_2 = torch.clone(population)
    random_indices_population = torch.randperm(POPULATION_SIZE, device=device)
    population_2 = population_2[random_indices_population, :]
    # create mask for picking between populations
    population_crossover_mask = torch.rand(POPULATION_SIZE, KNAPSACK_SIZE, device=device).lt_(0.5).to(dtype=torch.bool)
    # apply mask to pick items from the first population
    population.bitwise_and_(population_crossover_mask)
    # negate the mask
    population_crossover_mask.bitwise_not_()
    # apply again on the other population
    population_2.bitwise_and_(population_crossover_mask)
    # merge together
    population.bitwise_or_(population_2)

    # prepare the mask
    population_mutation_mask = torch.rand(POPULATION_SIZE, KNAPSACK_SIZE, device=device).lt_(mutation_chance).to(dtype=torch.bool)
    # apply mutation
    population.bitwise_xor_(population_mutation_mask)
    # transfer best part of the population
    population[:best_percent_amount] = population_best_ten_percent

############################################################
#
#  Correct non-admissible solutions
#  - first shuffles all items in all tensors accordingly using a random index permutation tensor to avoid bias - items with higher index have higher chance to be dropped
#  - calculates cumulative sum of weights and creates a mask to apply on population to remove excessive items
#
############################################################

random_indices_knapsack = torch.randperm(KNAPSACK_SIZE, device=device)
# shuffle items
knapsack_weights = knapsack_weights[random_indices_knapsack]
knapsack_prices = knapsack_prices[random_indices_knapsack]
population = population[:, random_indices_knapsack]

item_mask = torch.mul(population, knapsack_weights).cumsum_(1).le_(max_knapsack_weight)
population.bitwise_and_(item_mask)

############################################################
#
#  Extract best solution
#  - extracts and prints current best solution
#
############################################################

end = timer() - start
print("Timer ends...")
print("Tensor Genetic Algorithm took %f seconds on {}".format("CPU" if device.__str__() == "cpu" else torch.cuda.get_device_name()) % end)
print("\nBest found solution:")
print(torch.max(torch.mul(population, knapsack_prices).cumsum_(1)[:, KNAPSACK_SIZE - 1]))
print("\nTarget solution:")
print(target_solution)
print("\nTarget ceil:")
print(target_ceil)

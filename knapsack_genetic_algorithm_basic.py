import math
import random
import time
from timeit import default_timer as timer


# knapsack class, does not restrict the number of items that can be put in, just its max weight
class Knapsack:
    def __init__(self, item_prices_array: list, item_weights_array: list, max_weight: int):
        self.item_prices_array = item_prices_array
        self.item_weights_array = item_weights_array
        self.max_weight = max_weight


class Solution:
    def __init__(self, solution_array: list, knapsack_data: Knapsack):
        self.solution_array = solution_array
        self.knapsack_data = knapsack_data
        self.fitness = self.calculate_fitness()

    # higher is better

    def calculate_fitness(self):

        return_value = 0

        for x in range(len(self.solution_array)):
            return_value += self.knapsack_data.item_prices_array[x] if self.solution_array[x] else 0

        return return_value

    def calculate_weight(self):
        weight = 0

        for x in range(len(self.solution_array)):
            weight += self.knapsack_data.item_weights_array[x] if self.solution_array[x] else 0

        return weight

    # true if admissible, else otherwise
    def is_admissible(self):
        return self.calculate_weight() <= self.knapsack_data.max_weight

    def fix_solution(self):
        not_admissible = True

        while not_admissible:
            start_index = random.randint(0, len(self.solution_array) - 1)

            while not self.solution_array[start_index]:
                start_index = (start_index + 1) % len(self.solution_array)

            self.solution_array[start_index] = False
            not_admissible = not self.is_admissible()

        self.fitness = self.calculate_fitness()

    # mates this and other solution, creating a new one
    def mate(self, solution2):
        new_solution_array = []

        for x in range(len(self.solution_array)):
            rand = random.random()

            if rand < 0.5:
                new_solution_array.append(self.solution_array[x])
            else:
                new_solution_array.append(solution2.solution_array[x])

        if random.random() < 0.8:
            index = random.randint(0, len(self.solution_array) - 1)
            new_solution_array[index] = ~new_solution_array[index]

        return Solution(new_solution_array, self.knapsack_data)


# generates a knapsack based on the size of elements
def generate_knapsack(size: int):
    item_prices_array = []
    item_weights_array = []

    min_item_weight = math.log10(size) * math.pow(size, 1 / 4)
    max_item_weight = math.log10(size) * math.pow(size, 1 / 2)
    min_item_price = 2
    max_item_price = 100

    max_knapsack_weight = round((min_item_weight + max_item_weight) / (random.uniform(3, 5)) * size)

    for x in range(size):
        item_prices_array.append(random.randint(min_item_price, max_item_price))
        item_weights_array.append(round(random.uniform(min_item_weight, max_item_weight)))

    return Knapsack(item_prices_array, item_weights_array, max_knapsack_weight)


# generates initial population of solutions of defined size and knapsack
def generate_initial_population(size: int, knapsack: Knapsack):
    knapsack_size = len(knapsack.item_prices_array)
    population = []

    for x in range(size):
        solution_array = [False] * knapsack_size
        solution_weight = 0
        items_put = round(knapsack_size / 8)

        for y in range(items_put):
            index_pick = random.randint(0, knapsack_size - 1)

            if not solution_array[index_pick] and solution_weight + knapsack.item_weights_array[index_pick] <= knapsack.max_weight:
                solution_array[index_pick] = True
                solution_weight += knapsack.item_weights_array[index_pick]
            else:
                break

        population.append(Solution(solution_array, knapsack))

    return population


def get_target_solution(knapsack: Knapsack):
    solution_length = len(knapsack.item_prices_array)
    indices = []
    target_solution = [False] * solution_length

    for index in range(solution_length):
        indices.append(index)

    coefficient_tuples = zip(knapsack.item_prices_array, knapsack.item_weights_array, indices)
    coefficient_tuples = sorted(coefficient_tuples, key=lambda x: x[0] / x[1], reverse=True)

    weight_left = knapsack.max_weight
    first_time_less = True
    final_price = 0

    for price, weight, index in coefficient_tuples:
        if weight_left < weight:
            if first_time_less:
                final_price += (weight_left / weight) * price
                first_time_less = False

            continue

        weight_left -= weight
        target_solution[index] = True
        final_price += price

    return Solution(target_solution, knapsack), final_price


def main():
    random.seed()
    knapsack_size = 1000
    population_size = 5000
    max_generations = 1000

    knapsack = generate_knapsack(knapsack_size)
    population = generate_initial_population(population_size, knapsack)
    target = get_target_solution(knapsack)
    output_file = open("output/naive_output_knapsack.csv", "a")
    current_best = 0
    start = timer()

    for x in range(max_generations):
        population = sorted(population, key=lambda x1: x1.fitness, reverse=True)
        new_generation = []

        best_ten_percent_amount = round(10 * population_size / 100)
        new_generation.extend(population[:best_ten_percent_amount])

        to_be_mated_amount = population_size - best_ten_percent_amount

        for y in range(to_be_mated_amount):
            best_fifty_percent_amount = round(50 * population_size / 100)
            parent1 = random.choice(population[:best_fifty_percent_amount])
            parent2 = random.choice(population[:best_fifty_percent_amount])
            child = parent1.mate(parent2)

            if not child.is_admissible():
                child.fix_solution()

            new_generation.append(child)

        population = new_generation

        if population[0].fitness > current_best:
            current_best = population[0].fitness
            print("Generation: {}, Best fitness: {}, Target estimate: {}, Target ceil: {}".format(x, population[0].fitness, target[0].fitness, target[1]))

        # if population[0].fitness > target[0].fitness:
        #     break

    duration = timer() - start
    output_file.write("naive;{};{};{};{}\n".format(population_size, knapsack_size, max_generations, duration))
    output_file.close()
    print("Simple Genetic Algorithm took %f seconds" % duration)


if __name__ == '__main__':
    main()

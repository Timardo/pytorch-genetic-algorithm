import random
from timeit import default_timer as timer
import torch
import cpuinfo

############################################################
#
#  Genetic algorithm implementation using tensors with goal to solve bus count optimization problem
#  Version: 1.1
#
#  Known issues:
#  - mutation and crossover operator produce non-admissible solutions
#  - operation for evaluating the level of inadmissibility is memory intensive and unusable on larger problems or populations
#
############################################################

with torch.no_grad():
    ############################################################
    #
    #  Prepare constants
    #
    ############################################################

    print("\nSetting up environment...")
    # specifies the problem to solve from <0;10>, 0 is a dummy problem meant for testing with readable output, others are real data
    problem = "2"
    # specifies the population size to use
    population_size = 20000
    # population must be divisible by 2, fix it just in case
    population_size += population_size % 2
    # number of generations
    max_generations = 1000
    # initial number of buses to use in initial population
    starting_buses_count = 10
    # amount of individuals from a population to pick and transfer to new population
    best_percent_amount = int(population_size * 0.1)
    # 50% of individuals from a population
    percent_to_mate_amount = int(population_size * 0.5)
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

    print("\nLoading data...")
    csv_bus_connections_file = open("input/spoje_id_DS{}_J_1.csv".format(problem), "r")
    # ignore csv header line
    csv_bus_connections_file.readline()
    csv_bus_connections_file_lines = csv_bus_connections_file.readlines()
    csvDistMatrixFile = open("input/Tij_DS{}_J_1.csv".format(problem), "r")
    # load the number of bus connections this matrix consists of
    bus_connection_count = int(csvDistMatrixFile.readline())
    # this should be ALWAYS the same as bus_connection_count, ignore
    csvDistMatrixFile.readline()

    start_times_vec = torch.tensor([int(connection.split(";")[6]) for connection in csv_bus_connections_file_lines if len(connection) > 0], device="cpu")
    end_times_vec = torch.tensor([int(connection.split(";")[7]) for connection in csv_bus_connections_file_lines if len(connection) > 0], device="cpu")
    dist_matrix = torch.tensor([[int(value) for value in csvDistMatrixFile.readline().split(";") if value != "\n"] for j in range(bus_connection_count)], device="cpu")
    transition_matrix_negated = end_times_vec.unsqueeze(-1).expand(bus_connection_count, bus_connection_count).add(dist_matrix).gt(start_times_vec).to(device=device)

    # serve as a static index matrices for creating list of solution matrices from a population in order to get the fitness of solutions (number of buses)
    population_indices = torch.arange(population_size, device=device).unsqueeze(-1).expand(population_size, bus_connection_count)
    row_indices = torch.arange(bus_connection_count, device=device)
    # a mirror of population representing each solution as a matrix mask with 1 where bus index and bus connection index meet
    population_matrix = torch.zeros(population_size, bus_connection_count, starting_buses_count, device=device, dtype=torch.int16)
    # chance to mutate a single element (change the bus that is serving a specific bus lane to a random bus that already has some bus lanes)
    # base chance to mutate a solution is 80%, applied globally must be divided by number of bus lanes
    # THIS DOES NOT GUARANTEE A MAXIMUM OF ONE MUTATION PER SOLUTION!
    mutation_chance = 0.8 / bus_connection_count
    # indices matrix used to create 4D matrix that defines current transitions between bus connections
    population_indices_1 = torch.arange(population_size, device=device).unsqueeze(-1).unsqueeze(-1).expand(population_size, starting_buses_count, bus_connection_count).int()
    # indices matrix used to create 4D matrix that defines current transitions between bus connections
    row_indices_1 = torch.arange(starting_buses_count, device=device).unsqueeze(-1).expand(starting_buses_count, bus_connection_count).int()

    ############################################################
    #
    #  Create initial population
    #
    ############################################################

    print("\nGenerating initial population...")
    population_matrix_basic = [[0 for j in range(bus_connection_count)] for i in range(population_size)]
    buses_last_bus_lane = [[-1 for j in range(starting_buses_count)] for i in range(population_size)]

    for i in range(population_size):
        if i % (population_size / 10) == 0:
            print("{}%".format((i / population_size) * 100))

        for j in range(bus_connection_count):
            rand_bus = random.randint(0, starting_buses_count - 1)
            starting_bus = rand_bus
            is_valid = buses_last_bus_lane[i][rand_bus] == -1 or end_times_vec[buses_last_bus_lane[i][rand_bus]] + dist_matrix[buses_last_bus_lane[i][rand_bus], j] <= start_times_vec[j]

            while not is_valid:
                rand_bus = (rand_bus + 1) % starting_buses_count
                is_valid = buses_last_bus_lane[i][rand_bus] == -1 or end_times_vec[buses_last_bus_lane[i][rand_bus]] + dist_matrix[buses_last_bus_lane[i][rand_bus], j] <= start_times_vec[j]

                if not is_valid and starting_bus is rand_bus:
                    print("Cannot generate starting population with {} starting buses. Consider increasing the amount of starting buses to lower the chance of this happening.".format(starting_buses_count))
                    exit(1)

            population_matrix_basic[i][j] = rand_bus
            buses_last_bus_lane[i][rand_bus] = j

    # represents population, one row is one solution, one column is one bus lane with the number indicating which bus is serving this lane
    population = torch.tensor(population_matrix_basic, device=device)

    ############################################################
    #
    #  Calculate fitness function
    #  - this version of the fitness function takes into account the level of inadmissibility of solutions of population
    #  - each incorrect (impossible) transition between two bus connections counts as +1 to fitness function which means one more bus would be needed for a valid solution
    #  - fixing an inadmissible solution is just as complex as using a mutation operator that produces admissible solutions only
    #  - this is just a proof of concept that is not usable on large problems with large populations due to insanely high memory requirements
    #  - as an example, the largest problem with around 1000 bus lanes and 100 starting buses can be run with maximum POPULATION_SIZE of 86 on a system with 64GB RAM and 12GB GPU VRAM
    #    using ~43GB of shared GPU memory out of 44GB available (using more memory than the GPU can provide dramatically slows down the process anyway and should be avoided at all costs)
    #
    ############################################################

    population_matrix.mul_(0)
    # set 1 to indices that indicate which buses serve which bus connection
    population_matrix[population_indices, row_indices, population] = 1
    # transposed population matrix for ease of indexing
    population_matrix_transposed = population_matrix.transpose(1, 2)
    ############################################################
    # Old code
    ############################################################
    # final mask containing all current bus connection transitions filtered to upper part of diagonal (diagonal-exclusive)
    # the size of this mask is POPULATION_SIZE * STARTING_BUSES_COUNT * bus_connection_count * bus_connection_count, the largest problem is 1000 bus connections and a minimum of ~60 buses
    # this makes this mask the size of 60_000_000 * POPULATION_SIZE bytes which is about 60MB per individual in population which is not viable for large population sizes
    # TODO: find an alternative that does not take as much space
    # population_matrix_transposed_mask = population_matrix_transposed[population_indices_1, row_indices_1].bitwise_and(population_matrix_transposed[population_indices_1, row_indices_1].transpose(2, 3)).triu(diagonal=1)
    # current number of transitions in population
    # population_current_transitions = population_matrix_transposed_mask.cumsum(2).cumsum(3)[:, :, bus_connection_count - 1, bus_connection_count - 1].cumsum(1)[:, STARTING_BUSES_COUNT - 1]
    # current transitions filtered with possible transitions mask omitting impossible transitions
    # population_possible_transitions = population_matrix_transposed_mask.bitwise_and(transition_matrix).cumsum(2).cumsum(3)[:, :, bus_connection_count - 1, bus_connection_count - 1].cumsum(1)[:, STARTING_BUSES_COUNT - 1]
    # the difference between the number of current transitions and possible transitions, anything above zero is a non-admissible solution
    # population_transition_difference = population_current_transitions.add_(population_possible_transitions.mul_(-1))
    # get fitness tensor (number of buses) by summing the number of server bus connections for each bus, then summing the number of non-zero-bus-connection-serving buses and extracting this number
    # fitness = population_matrix.cumsum_(1).gt_(0).cumsum_(2)[:, bus_connection_count - 1, STARTING_BUSES_COUNT - 1]
    # add the difference to fitness
    # fitness.add_(population_transition_difference)
    ############################################################
    # New code that does not use temporary structures but rather does everything in one connection to save space and increase performance
    ############################################################
    fitness = population_matrix_transposed[population_indices_1, row_indices_1].bitwise_and_(population_matrix_transposed[population_indices_1, row_indices_1].transpose_(2, 3)).triu_(diagonal=1).bitwise_and_(transition_matrix_negated).cumsum_(2).cumsum_(3)[:, :, bus_connection_count - 1, bus_connection_count - 1].cumsum_(1)[:, starting_buses_count - 1].add_(population_matrix.cumsum_(1).gt_(0).cumsum_(2)[:, bus_connection_count - 1, starting_buses_count - 1])

    print("\nStarting best solution: {}".format(fitness.sort(0, descending=False)[0][0].item()))
    print("\nTimer starts...")
    start = timer()

    for x in range(max_generations):
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
        #  - uses k-point crossover where k is the number of bus connections
        #
        ############################################################

        # extract best part of the population to separate tensor
        population_best_ten_percent = population[:best_percent_amount]
        # initialize random indices tensor and shuffle the rows of the population for more random crossover
        population_shuffle_indices = torch.randperm(population_size, device=device)
        # duplicate best half of the population over the worst half
        population[percent_to_mate_amount:] = population[:percent_to_mate_amount]
        # clone population to create a pair for each solution
        population_2 = torch.clone(population)
        # shuffle both populations
        population = population[population_shuffle_indices, :]
        population_shuffle_indices = torch.randperm(population_size, device=device)
        population_2 = population_2[population_shuffle_indices, :]
        # create mask for picking between populations
        population_crossover_mask = torch.rand(population_size, bus_connection_count, device=device).lt_(0.5).bool()
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
        population_mutation_mask = torch.rand(population_size, bus_connection_count, device=device).lt_(mutation_chance).bool()
        # clear population matrix
        population_matrix.mul_(0)
        # set 1 to indices that indicate which buses serve which bus connection
        population_matrix[population_indices, row_indices, population] = 1
        # sort population matrix in a way to get the count of buses for each solution as well as the possible bus indices for each solution
        # each possible_indices[i] vector's first X values where X is represented by buses_counts[i] represent the possible bus indices
        # that the solution can mutate to without changing the fitness (not mutating to empty buses)
        buses_counts, mutated_bus_matrix = population_matrix.sort(1, descending=True)[0][:, 0].sort(1, descending=True)
        # create index matrix to use for multiplying random float values to get random indices for bus selection
        # max_bus_indices = buses_counts.cumsum_(1)[:, starting_buses_count - 1].unsqueeze(-1).expand(population_size, bus_connection_count)
        # create indices to use for mapping to possible buses
        population_mutation_indices = torch.rand(population_size, bus_connection_count, device=device).mul_(buses_counts.cumsum_(1)[:, starting_buses_count - 1].unsqueeze(-1).expand(population_size, bus_connection_count)).int()
        # finally represents a new completely random population filtered with the population_mutation_mask
        mutated_bus_matrix = mutated_bus_matrix[population_indices, population_mutation_indices].mul_(population_mutation_mask)
        # mutated_bus_matrix = torch.rand(POPULATION_SIZE, bus_connection_count, device=device).mul_(STARTING_BUSES_COUNT).mul_(population_mutation_mask).int()
        # flip the mask
        population_mutation_mask.bitwise_not_()
        # apply the mask on population and combine with the mutation population
        population.mul_(population_mutation_mask).add_(mutated_bus_matrix)

        ############################################################
        #
        #  Calculate fitness function
        #  - this version of the fitness function takes into account the level of inadmissibility of solutions of population
        #  - each incorrect (impossible) transition between two bus connections counts as +1 to fitness function which means one more bus would be needed for a valid solution
        #  - fixing an inadmissible solution is just as complex as using a mutation operator that produces admissible solutions only
        #  - this is just a proof of concept that is not usable on large problems with large populations due to insanely high memory requirements
        #  - as an example, the largest problem with around 1000 bus lanes and 100 starting buses can be run with maximum POPULATION_SIZE of 86 on a system with 64GB RAM and 12GB GPU VRAM
        #    using ~43GB of shared GPU memory out of 44GB available (using more memory than the GPU can provide dramatically slows down the process anyway and should be avoided at all costs)
        #
        ############################################################

        population_matrix.mul_(0)
        # set 1 to indices that indicate which buses serve which bus connection
        population_matrix[population_indices, row_indices, population] = 1
        # transposed population matrix for ease of indexing
        population_matrix_transposed = population_matrix.transpose(1, 2)
        ############################################################
        # Old code
        ############################################################
        # final mask containing all current bus connection transitions filtered to upper part of diagonal (diagonal-exclusive)
        # the size of this mask is POPULATION_SIZE * STARTING_BUSES_COUNT * bus_connection_count * bus_connection_count, the largest problem is 1000 bus connections and a minimum of ~60 buses
        # this makes this mask the size of 60_000_000 * POPULATION_SIZE bytes which is about 60MB per individual in population which is not viable for large population sizes
        # TODO: find an alternative that does not take as much space
        # population_matrix_transposed_mask = population_matrix_transposed[population_indices_1, row_indices_1].bitwise_and(population_matrix_transposed[population_indices_1, row_indices_1].transpose(2, 3)).triu(diagonal=1)
        # current number of transitions in population
        # population_current_transitions = population_matrix_transposed_mask.cumsum(2).cumsum(3)[:, :, bus_connection_count - 1, bus_connection_count - 1].cumsum(1)[:, STARTING_BUSES_COUNT - 1]
        # current transitions filtered with possible transitions mask omitting impossible transitions
        # population_possible_transitions = population_matrix_transposed_mask.bitwise_and(transition_matrix).cumsum(2).cumsum(3)[:, :, bus_connection_count - 1, bus_connection_count - 1].cumsum(1)[:, STARTING_BUSES_COUNT - 1]
        # the difference between the number of current transitions and possible transitions, anything above zero is a non-admissible solution
        # population_transition_difference = population_current_transitions.add_(population_possible_transitions.mul_(-1))
        # get fitness tensor (number of buses) by summing the number of server bus connections for each bus, then summing the number of non-zero-bus-connection-serving buses and extracting this number
        # fitness = population_matrix.cumsum_(1).gt_(0).cumsum_(2)[:, bus_connection_count - 1, STARTING_BUSES_COUNT - 1]
        # add the difference to fitness
        # fitness.add_(population_transition_difference)
        ############################################################
        # New code that does not use temporary structures but rather does everything in one connection to save space and increase performance
        ############################################################
        fitness = population_matrix_transposed[population_indices_1, row_indices_1].bitwise_and_(population_matrix_transposed[population_indices_1, row_indices_1].transpose_(2, 3)).triu_(diagonal=1).bitwise_and_(transition_matrix_negated).cumsum_(2).cumsum_(3)[:, :, bus_connection_count - 1, bus_connection_count - 1].cumsum_(1)[:, starting_buses_count - 1].add_(population_matrix.cumsum_(1).gt_(0).cumsum_(2)[:, bus_connection_count - 1, starting_buses_count - 1])

    ############################################################
    #
    #  Sort population
    #
    ############################################################

    # sort population according to the fitness function
    sort_result, sort_indices = fitness.sort(0, descending=False)
    population = population[sort_indices]
    fitness = sort_result

    ############################################################
    #
    #  Print result
    #
    ############################################################

    duration = timer() - start
    print("Timer ends...")
    device_name = cpuinfo.get_cpu_info()['brand_raw'] if device.__str__() == "cpu" else torch.cuda.get_device_name()
    print("\nTensor Genetic Algorithm took %f seconds" % duration)
    print("Device: {}".format(device_name))
    print("Population: {}".format(population_size))
    print("Generations: {}".format(max_generations))
    print("Problem number: {}".format(problem))
    print("Best found solution: {}".format(fitness[0].item()))
    print(population[0])
    output_file = open("output/tensor_output_bus_gpu_only.csv", "a")
    output_file.write("{},{},{},{},{},{}\n".format(device.__str__(), population_size, problem, max_generations, starting_buses_count, duration))

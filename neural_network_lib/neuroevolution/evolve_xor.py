import numpy as np
import neural_nets as nn


def generate_population(population_size, nn_architecture):
    """
    Generate of population of neural_nets of size n using the passed nn architecture.

    Input:
        n (Int) - The size of the population to create.
        nn_architecture (list of dicts) - list of dicts where each dict specifies the inputs to the NeuralNet.add_layer
                                          method. If an input is not specified in the dict, the default will be used.
    Output:
        population (list of NeuralNets) - A list holding the members of the population created.

    """
    population = []
    for _ in range(population_size):
        population.append(nn.create_nn_from_arch(nn_architecture))

    return population


def compute_tot_fitness(fitness_function, pop):
    """
    Computes the total fitness of the population which is the sum of the fitness values of every
    element of the population.

    Input:
        fitness_function (function) - Function to determine the fitness of a member of the population
        pop (list of population) - The list containing the population of chromosomes.

    Output:
        (total_fitness, best_fitness, best_member) (float, float, pop element) - The total fitness of the population.

    """
    probs = np.zeros(len(pop))  # list to house probabilites
    best_member = ''
    best_fitness = -10**18
    total_fitness = 0  # The sum of of all the fitness values from the population.
    for i, chromosome in enumerate(pop):
        new_fitness = fitness_function(chromosome)
        if new_fitness > best_fitness:
            best_member = chromosome
            best_fitness = new_fitness
        total_fitness += new_fitness
        probs[i] = new_fitness
    probs = probs / total_fitness
    return total_fitness, best_fitness, best_member, probs


def crossover(NN1, NN2, p_c, p_m):
    """
    Perform the crossover for the two chromosomes x and y.

    Input:
        NN1, NN2 (members of the population) - Members of the population.
        p_c (float) - The crossover rate (number between 0 and 1).
        p_m (float) - The mutation rate (number between 0 and 1).

    Output:
        child (type of pop members) - The child from the mating process.

    """
    if np.random.choice([0, 1], p=[1-p_c, p_c]):
        return nn.mate_neural_nets(NN1, NN2, p_m)
    else:
        return np.random.choice([NN1, NN2])


def evolve_generation(pop, probs, best_member, p_c, p_m):
    """
    Create a new generation by running the selection and crossover process and
    then mutate_population.

    Input:
        pop (list) - The list housing the current population.
        fitness_function (function) - The function to compute the fitness of a member of the population.
        p_c - float - The crossover rate (float between 0 and 1).
        p_m (float) - The mutation rate (float between 0 and 1).

    Output:
        new_pop - list - List of the new population evolved from the input population.
    """
    if best_member is None:
        new_pop = []
    else:
        new_pop = [best_member]
    while len(new_pop) < len(pop):
        NN1, NN2 = np.random.choice(pop, size=2, p=probs)
        new_pop.append(crossover(NN1, NN2, p_c, p_m))
    return new_pop


def main(fitness_function, nn_architecture, population_size=4, p_c=0.9, p_m=0.01, generations=10000):
    """
    The main function which runs the evolution of the population.

    Input:
        fitness_function (function) - A function with which fitness is computed.
        nn_architecture (list of dicts) - list of dicts where each dict specifies the inputs to the NeuralNet.add_layer
                                          method. If an input is not specified in the dict, the default will be used.
        population_size (int) - The size of the population to create.
        p_c (float) - The crossover rate
        p_m (float) - The mutation rate
        generations (int) - The number of generations to run

    Returns:
        None

    """
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    pop = generate_population(population_size, nn_arch)
    i = 0
    total_fitness = 0
    best_fitness = 0
    j = 0
    while i < generations:
        total_fitness, best_fitness, best_member, probs = compute_tot_fitness(fitness_function, pop)
        if i % (generations // 10) == 0:
            # print("Population {}: {}".format(i, pop))
            print(f"EPOCH {j}")
            print(f"Average fitness is {total_fitness / population_size}.")
            print(f"Best fitness is {best_fitness}\n.")
            print(f"Best member gives \n{best_member.feed_forward(x)}\n.")
            print(f"LAYER 0:\n  WEIGHTS=\n{best_member.layers[0].weights}\n  BIAS=\n{best_member.layers[0].bias}")
            print(f"LAYER 1:\n  WEIGHTS=\n{best_member.layers[1].weights}\n  BIAS=\n{best_member.layers[1].bias}")
            j += 1
        pop = evolve_generation(pop, probs, best_member, p_c, p_m)
        i += 1
    print(f"EPOCH {j}")
    print(f"Average fitness is {total_fitness / population_size}.")
    print(f"Best fitness is {best_fitness}\n.")    
    print(f"Best member gives \n{best_member.feed_forward(x)}\n.")
    print(f"LAYER 0:\n  WEIGHTS=\n{best_member.layers[0].weights}\n  BIAS=\n{best_member.layers[0].bias}")
    print(f"LAYER 1:\n  WEIGHTS=\n{best_member.layers[1].weights}\n  BIAS=\n{best_member.layers[1].bias}")





layer_1 = dict(
        layer_type='Dense',
        num_input_units=2,
        num_output_units=2,
        activation_function='ReLu',
        st_dev=2
        )
layer_2 = dict(
        layer_type='Dense',
        num_input_units=2,
        num_output_units=1,
        activation_function='identity',
        st_dev=2
        )
nn_arch = [layer_1, layer_2]


def fitness(NN):
    """
    Compute the fitness of the NeuralNet.

    """
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    expected_y = np.array([[0], [1], [1], [0]])
    y = NN.feed_forward(x)
    error = expected_y - y
    return 1 / (np.square(np.dot(error.T, error)).squeeze() + 0.01)


if __name__ == '__main__':
    main(fitness_function=fitness, nn_architecture=nn_arch, population_size=100, p_c=0.9, p_m=0.05, generations=5000)

# =============================================================================
# layer_1 = dict(
#         layer_type='Dense',
#         num_input_units=2,
#         num_output_units=2,
#         activation_function='ReLu',
#         weights=np.array([[-26.29607697, 0.70982854], [0.93518267, -12.67239673]]),
#         bias=np.array([[-0.01315423, -0.01260735]])
#         )
# layer_2 = dict(
#         layer_type='Dense',
#         num_input_units=2,
#         num_output_units=1,
#         activation_function='identity',
#         weights=np.array([[1.43807166, 1.0819197]]),
#         bias=np.array([[0.00362729]])
#         )
# nn_arch = [layer_1, layer_2]
# NN = nn.create_nn_from_arch(nn_arch)
# 
# np.array([[-26.29607697, 0.70982854], [0.93518267, -12.67239673]])
# np.array([[-0.01315423, -0.01260735]])
# np.array([[1.43807166, 1.0819197]])
# np.array([[0.00362729]])
# =============================================================================


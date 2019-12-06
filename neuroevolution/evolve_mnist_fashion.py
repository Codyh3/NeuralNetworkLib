import numpy as np
import neural_nets as nn
from tensorflow import keras


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
    pop = generate_population(population_size, nn_arch)
    i = 0
    total_fitness = 0
    best_fitness = 0
    j = 0
    while (best_fitness / 60000) < 0.99:  # i < generations:
        total_fitness, best_fitness, best_member, probs = compute_tot_fitness(fitness_function, pop)
        if i % 10 == 0:  # i % (generations // 100) == 0:
            # print("Population {}: {}".format(i, pop))
            print(f"EPOCH {j}")
            print(f"Average fitness is {total_fitness / population_size}.")
            print(f"Best fitness is {best_fitness}\n.")
            j += 1
        pop = evolve_generation(pop, probs, best_member, p_c, p_m)
        i += 1
    print(f"EPOCH {j}")
    print(f"Average fitness is {total_fitness / population_size}.")
    print(f"Best fitness is {best_fitness}\n.")



layer_1 = dict(
        layer_type='Dense',
        num_input_units=784,
        num_output_units=128,
        activation_function='ReLu',
        st_dev=2
        )
layer_2 = dict(
        layer_type='Dense',
        num_input_units=128,
        num_output_units=10,
        activation_function='identity',
        st_dev=2
        )
nn_arch = [layer_1, layer_2]
NN = nn.create_nn_from_arch(nn_arch)


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape((train_images.shape[0], -1))
test_images = test_images.reshape((test_images.shape[0], -1))


def fitness(NN):
    """
    Compute the fitness of the NeuralNet.

    """
    y = np.argmax(NN.feed_forward(train_images), axis=1)
    return (y == train_labels).sum()


if __name__ == '__main__':
    main(fitness_function=fitness, nn_architecture=nn_arch, population_size=100, p_c=0.9, p_m=0.05, generations=500)

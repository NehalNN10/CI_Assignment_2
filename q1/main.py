import matplotlib.pyplot as plt
import numpy as np
import random

def read_matrix(file_name):
    with open(file_name, 'r') as f:
        return [[int(x) for x in line.split()] for line in f]

facilities = []
locations = []

# Initial parameters - as described in PDF

alpha = 1
beta = 1
gamma = 0.8
no_of_ants = 20
no_of_iterations_convergence = 10

# For record keeping
records = {}

def calculate_probability_matrix(pheromone_matrix):
    no_of_facilities = len(pheromone_matrix)
    no_of_locations = len(pheromone_matrix[0])
    pMatrix = np.zeros((no_of_facilities, no_of_locations))
    for i in range(no_of_locations):
        for j in range(no_of_facilities):
            pMatrix[i][j] = pheromone_matrix[i][j] / sum(pheromone_matrix[i])

    cumulative_matrix = np.cumsum(pMatrix, axis=1)
    return cumulative_matrix

# def calculate_validity(candidate_matrix, verbose):
# ? what is verbose for?
# * to make things more explicit via print statements - debugging
def calculate_validity(candidate_matrix):
    # No facility should be assigned to more than one location
    # No location should be assigned to more than one facility
    pop_size = len(candidate_matrix)
    no_of_vars = len(candidate_matrix[0])

    no_of_uniques = [0] * pop_size

    for i in range(pop_size):
        no_of_uniques[i] = len(set(candidate_matrix[i]))

    validity_check = all([x == no_of_vars for x in no_of_uniques])
    if sum(validity_check) != pop_size:
        print('Invalid solution found')
        return False
    return True

def fitness(population, distance_matrix, flow_matrix):
    no_of_facilities = len(distance_matrix)
    no_of_locations = len(distance_matrix)

    pop_size = len(population)
    pop_fitness = [0] * pop_size

    for idx in range(len(population)):
        chromosome = population[idx]
        distanceXfrequency = np.zeroes((no_of_facilities, no_of_locations))
        for i in range(no_of_locations):
            for j in range(no_of_facilities):
                dist = distance_matrix[i][j]
                freq = flow_matrix[i][j]
                distanceXfrequency[i][j] = dist * freq
        pop_fitness[idx] = np.sum(distanceXfrequency)

    if pop_size > 1:
        temp = np.hstack((population, pop_fitness.reshape(-1, 1)))
        temp = temp[np.argsort(temp[:, no_of_locations])]
        sorted_population = temp[:, :no_of_locations]
        pop_fitness = temp[:, no_of_locations]
    
    return (pop_fitness, sorted_population)


def main():

    # TODO: experiment with differing values of alpha, beta, gamma, no_of_ants, max_iter

    distance_matrix = read_matrix('src/distance_matrix.txt')
    flow_matrix = read_matrix('src/flow_matrix.txt')

    no_of_facilities = len(distance_matrix)
    no_of_locations = len(distance_matrix)

    # Initialize the pheromone matrix
    pheromone_matrix = np.ones((no_of_facilities, no_of_locations))

    # Maximum Iterations
    max_iter = 100

    # Heuristic matrix
    heuristic_matrix = 1 / distance_matrix

    # Main loop

    for iteration in range(max_iter):

        # Initialize a matrix with random values
        random_matrix = np.random.rand(no_of_facilities, no_of_locations)

        # Initialize location->facilities and facility->locations matrices
        location_facility_matrix = np.zeros((no_of_locations, no_of_facilities))
        facility_location_matrix = np.zeros((no_of_facilities, no_of_locations))

        for ant in range(no_of_ants):

            temp_pheromone_matrix = pheromone_matrix.copy()

            for i in range(no_of_locations):
                cumulative_matrix = calculate_probability_matrix(temp_pheromone_matrix)

                for j in range(no_of_facilities):
                    if cumulative_matrix[i, j] > random_matrix[i, j]:
                        location_facility_matrix[ant, j] = i
                        facility_location_matrix[ant, i] = j

                        temp_pheromone_matrix[:, j] = 0
                        break
            
        # Check feasibility
        if not calculate_validity(location_facility_matrix):
            print('Invalid solution found in iteration', iteration)
            continue

        # Calculate the fitnesses of the solutions
        fitnesses, sorted_population = fitness(location_facility_matrix, distance_matrix, flow_matrix)
        records[iteration] = [fitnesses[0], np.average(fitnesses), sorted_population[0]] # Record the best, average and best solution
        best_solution = sorted_population[0]

        # Plot the heatmap of the layout here - will do later
        
        # Check for convergence
        if iteration > 2 * no_of_iterations_convergence and len(set(records[iteration - no_of_iterations_convergence:iteration])) == 1:
            print('Converged at iteration', iteration)
            break
        
        # Update the pheromone matrix
        best_node_update = pheromone_scaler * (fitnesses[0]/fitnesses[-1])
        other_nodes_update = 1 - gamma

        for i in best_solution:
            for j in range(no_of_facilities):
                if j == np.where(i == best_solution)[0][0]:
                    pheromone_matrix[i, j] += best_node_update
                else:
                    pheromone_matrix[i, j] *= other_nodes_update
    print("Loop ended at iteration", iteration)

    plot_results()


# Plot the results
def plot_results():
    plt.figure()
    plt.plot(records.keys(), [x[0] for x in records.values()], label='Best')
    plt.plot(records.keys(), [x[1] for x in records.values()], label='Average')
    plt.legend()
    plt.show()

# ! Observation - neither alpha, beta nor gamma is used in the code
# ! How can we use them to improve the results?

if __name__ == '__main__':
    main()
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

# For record keeping
records = {}

def calculate_probability_matrix(pheromone_matrix):
    

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


            # Initialize the ant's path
            path = []

            # Randomly select a facility
            facility = random.randint(0, no_of_facilities - 1)
            path.append(facility)

            # Update the location->facility and facility->location matrices
            for location in range(no_of_locations):
                location_facility_matrix[location][facility] += 1
                facility_location_matrix[facility][location] += 1

            # Update the pheromone matrix
            for i in range(len(path) - 1):
                pheromone_matrix[path[i]][path[i + 1]] += 1


# Plot the results
def plot_results():
    pass
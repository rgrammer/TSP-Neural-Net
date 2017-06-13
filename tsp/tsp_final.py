'''
Created on Apr 12, 2017

@author: Robert Grammer - rgramme2@jhu.edu
'''

import random
import math
import copy

#Main function where the city names and city distances are established, annealing process is called, and final path
#and distance are displayed to the user
def main():
    print("Solving the Traveling Salesman Problem with a Boltzmann Machine:")
    city_names=["A", "B", "C", "D", "E"]
    #Matrix of distances between cities
    distances=[[0, 10, 20, 5, 18], [10, 0, 15, 32, 10], [20, 15, 0, 25, 16], [5, 32, 25, 0, 35], [18, 10, 16, 35, 0]]
    neuron_matrix = create_neuron_matrix()
    annealed_neuron_matrix = anneal(neuron_matrix, distances)
    print("The final neuron matrix is as follows, columns are epochs and rows are cities: ")
    print(annealed_neuron_matrix[0])
    print(annealed_neuron_matrix[1])
    print(annealed_neuron_matrix[2])
    print(annealed_neuron_matrix[3])
    print(annealed_neuron_matrix[4])
    #Determine a printable version of the path based on the annealed neuron matrix
    path = ""
    num_cities = 0
    optimal_distance = 0
    prev_city = 0
    for epoch in range(0, 6):
        for curr_city in range(0, 5):
            if(annealed_neuron_matrix[curr_city][epoch] == 1):  
                path += city_names[curr_city]
                if(epoch == 0):
                    prev_city = curr_city
                else:
                    optimal_distance = optimal_distance + distances[prev_city][curr_city]
                prev_city = curr_city
                if(num_cities < len(city_names)):
                    path += " ->"
                num_cities = num_cities + 1  
    print("The suggested route for the salesman is: " + path)
    print("The distance of the route is: " + str(optimal_distance))
    
#The create_neuron_matrix function creates a 5x6 random binary matrix where each cell represents a neuron. This 
#data structure makes it easy to track the states of the neurons. Each column in the matrix represents one of the
#six epochs. Each row in the matrix represents a city, so ultimately we need to have one neuron in each row and one in 
#each column that is on while all others are off to represent a valid path for the salesman.
#Note that this function randomly chooses a start city and only one of the neurons in the first and last epochs are
#ever on because the salesman has to start and finish in that city
def create_neuron_matrix():
    #Create middle ephochs with random binary states
    neuron_matrix = [[0]*5]*6
    #Randomly choose start city
    start_city = random.randint(0, 4)
    first_and_last_epoch = [0, 0, 0, 0, 0]
    first_and_last_epoch[start_city] = 1
    second_to_fifth_epochs = [[random.randint(0, 1) for i in range(5)] for j in range(4)]
    neuron_matrix[0] = first_and_last_epoch 
    neuron_matrix[1] = second_to_fifth_epochs[0]
    neuron_matrix[2] = second_to_fifth_epochs[1]
    neuron_matrix[3] = second_to_fifth_epochs[2]
    neuron_matrix[4] = second_to_fifth_epochs[3]
    neuron_matrix[5] = first_and_last_epoch
    #Convert randomly assigned lists into list of lists
    neuron_matrix = list(zip(*neuron_matrix))
    neuron_matrix = [list(elem) for elem in neuron_matrix]
    return neuron_matrix

#The consensus function determines the overall consensus value of a neuron matrix. In order for the program to 
#properly determine the result for this problem, the consensus function has to be calculated with weight values 
#that inihibit invalid updates to the neuron matrix. TO do this, the consensus function looks for invalid update
#scenarios and will calculate the consensus value with very high weights to inhibit the potential for an invalid 
#matrix. Such scenarios include the salesman visitng previously visited cities or multiple cities in the same 
#epoch. 
def consensus_function(distances, neuron_matrix, x, y):
    consensus_value = 0
    weight = 0
    #Consensus value does not need to be calculated if the neuron is off, it will be 0 in that scenario
    if(neuron_matrix[y][x] == 1):
        #Check to see if any of the other neurons representing the city the current neuron is related to are on, if
        #so, return a weight of 500 to inhibit this neuron
        visited = False
        for previous in range(0, 6):
            if(previous != x and neuron_matrix[y][previous] == 1):
                visited = True
        if(visited):
            weight = 500
        #Check to see if any of the other neurons in the same epoch are on, if so return a weight of 500 to 
        #inhibit this neuron
        simultaneous = False
        for city in range(0, 5):
            if(city != y and neuron_matrix[city][x] == 1):
                #If there are other neurons in the same epoch that are on, compare the distances of the respective 
                #cities from the previously visited city to determine which neuron being on would give the salesman
                #a shorter journey. If there is another neuron in the epoch that is on and represents a shorter
                #journey for the salesman, then inhibit this neuron with a weight of 500 otherwise use the distance
                #from the last city as the weight
                for previous in range(0, 5):
                    if(neuron_matrix[previous][x-1] == 1):   
                        if(distances[city][previous] < distances[y][previous]):
                            simultaneous = True
        if(simultaneous):
            weight = 500
        #If none of the prohibited scenarios exist, use the city distances from all of the other 'on' neurons as te weight
        if(visited == False and simultaneous == False):
            for city in range(0, 5):
                if(neuron_matrix[city][x-1] == 1):
                    weight = weight + distances[city][y]
        
        current_neuron_state = neuron_matrix[y][x]
        #Calculate the consensus value with the current neuron state and the weights determined above
        consensus_value = 0.5*weight*current_neuron_state 
    return consensus_value 

#The sigmoid function simply takes the Consensus delta and the current temperature from the annealing process
#and returns the curret activation function  
def sigmoid_function(deltaConsensus, curr_temp):
    activation_value = 1/(1+ math.exp(deltaConsensus/curr_temp))
    return activation_value

#Randomly picks a neuron in the neuron matrix to update, note the first and last epochs are ignored
def pick_random_neuron():
    x = random.randint(1, 4)
    y = random.randint(0, 4)
    return x, y

#The annealing process takes place in this function. Parameters are the distance matrix and the initial random neuron matrix
def anneal(neuron_matrix, distances):
    #Establish annealing process variables such as the initial temperature and cooling rate 
    curr_temp = 100.0
    deltaConsensus = 0
    coolingRate = 0.99995
    absoluteTemp = 1
    print("Annealing...this may take a few moments...")
    while curr_temp > absoluteTemp:
        #Establish a candidate neuron matrix
        candidate_neuron_matrix = copy.deepcopy(neuron_matrix)
        #Pick a random candidate neuron to update
        x, y = pick_random_neuron()
        #Update the randomly chose neuron in the candidate matrix
        if(candidate_neuron_matrix[y][x] == 1):
            candidate_neuron_matrix[y][x] = 0
        else:
            candidate_neuron_matrix[y][x] = 1
        #Calculate the initial consensus value and the candidate consensus value
        consensus = consensus_function(distances, neuron_matrix, x, y)
        candidate_consensus = consensus_function(distances, candidate_neuron_matrix, x, y)
        #Calculate the consensus delta
        deltaConsensus = candidate_consensus - consensus
        #To promote valid city visits, convert the sign of the consensus value
        if(deltaConsensus <= 35 and deltaConsensus >= -35):
            deltaConsensus = deltaConsensus * -1
        #Randomly generate an acceptance criteria between 0 and 1
        acceptance_criteria = random.uniform(0, 1)
        #Generate an acceptance probability from the consensus delta and the current annealing temperature
        acceptance_probability = sigmoid_function(deltaConsensus, curr_temp)
        #If the acceptance criteria is less that than the acceptance probability, update the neuron to matrix with the 
        #candidate neuron
        if(acceptance_criteria < acceptance_probability):
            neuron_matrix[y][x] = copy.deepcopy(candidate_neuron_matrix[y][x])
            print("Updated neuron matrix: ")
            print(neuron_matrix[0])
            print(neuron_matrix[1])
            print(neuron_matrix[2])
            print(neuron_matrix[3])
            print(neuron_matrix[4])
        #Update the annealing process temperature
        curr_temp = curr_temp * coolingRate
       
    print("Annealing process completed!")
    return neuron_matrix 
if __name__ == '__main__':
    main()
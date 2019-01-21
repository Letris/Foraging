import numpy as np
from robobo import SimulationRobobo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time 
import copy
import pickle

class Controller:
    def __init__(self, inputsize, outputsize, hiddensize):
        #parameters
        self.inputSize = inputsize
        self.outputSize = outputsize
        self.hiddenSize = hiddensize
        self.fitness = 0

        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (YxZ) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (YxZ) weight matrix from hidden to output layer

    def forward(self, X):
        #forward propagation through our network
        self.z = np.dot(X, self.W1) # dot product of X (input) and first set of YxZ weights
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of YxZ weights
        o = self.sigmoid(self.z3) # final activation function
        return o

    def sigmoid(self, s):
        # activation function
        return 1/(1+np.exp(-s))

    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")

    def predict(self, state):
        return self.forward(state)

    def mutate_weights(self, sigma):
        # sample random mutations from a gaussian distribution
        W1_mutations = numpy.random.normal(0, sigma, (self.inputSize, self.hiddenSize))
        W2_mutations = numpy.random.normal(0, sigma, (self.hiddenSize, self.outputSize))

        # mutate the weights with the sampled mutations
        self.W1 = [[self.W1[i][j] + W1_mutations[i][j]  for j in range(len(self.W1[0]))] for i in range(len(self.W1))]
        self.W2 =  [[self.W1[i][j] + W1_mutations[i][j]  for j in range(len(self.W1[0]))] for i in range(len(self.W1))]

    def return_weights(self):
        return self.W1, self.W2

    def update_fitness(self, state, action):
        # ToDo

    def save(self):
        with open("brain.file", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


def train():
    # define neural network
    input_size = 0
    output_size = 0
    hidden_size = 0

    # initialize neural network with random weights
    champion = Controller(input_size, output_size, hidden_size) # inputsize, outputsize, hiddensize

    # define other parameters
    M = 0 # amount of episodes
    S = 0 # amount of steps per episode
    sigma = 0.01 # standard deviation
    min_sigma = 0.01 # minimal standard deviation

    # save all controllers
    controllers = [champion]
    
    for s in range(S):
        # Reset environment 
        env = SimulationRobobo().connect(address='192.168.1.135', port=19997)
        time.sleep(3)
        env.reset()
        time.sleep(10)

        # create challenger
        challenger = copy.deepcopy(champion).mutate_weights(sigma)
        controllers.append(challenger)

        # Test challenger
        for s in range(S):
            # Obtain state
            state = env.obtain_state
            # Decide action
            action = NN.predict(state)
            # Update fitness 
            challenger.update_fitness()
            # Perform action
            env.move(action[0], action[1])

        # Determine whether challenger is better than current champion
        if challenger.fitness > champion.fitness:
            champion = challenger
            sigma = min_sigma
        else:
            sigma *= 2
        
        # stop current environment
        env.stop_world()
        time.sleep(10)

    # plot fitness over time
    plot_fitness(controllers)
    # save best controller
    champion.save()

    if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        # model.save('/Users/Tristan/Downloads/Uni/LearningMachines/brain.hdf5')
        print('Exiting.')
    
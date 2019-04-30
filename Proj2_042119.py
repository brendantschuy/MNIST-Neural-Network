import csv
import random
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

#For debugging purposes, allows unabridged printing of arrays.
np.set_printoptions(threshold=sys.maxsize)
warnings.filterwarnings("error")

class Classifier(object):
    #initialize the classifier
    def __init__(self, num_outputs, learning_rate, num_inputs, num_hidden, momentum, epochs):
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = 10
        self.momentum = momentum
        self.epochs = epochs

    def sigmoid(self, val):
        #try/except was to see what was happening with momentum = 1
        try:
            ret = 1 / (1 + np.exp(val * -1))
        except:
            print("overflow error with val = ", val)
            ret = 1
        return ret


    #read in the input and organize it in data structures (matrices/numPy arrays)
    def read(self, fileName):
        self.inputFile = open(fileName, encoding='utf-8-sig')

        with self.inputFile as file:
            self.inputMatrix = [[float(digit) for digit in line.split(',')] for line in file]

        self.data = np.asarray(self.inputMatrix)
        self.data[:,1:] = self.data[:,1:]/255

        #not shuffling while debugging:
        np.random.shuffle(self.data)

        #bias value is always 1:
        self.inputBiasValues = np.full((len(self.data), 1), 1)

        #randomize initial weights:
        #input to hidden weights:
        self.inputToHiddenWeights = np.random.uniform(-0.05, 0.05, (self.num_hidden, self.num_inputs + 1))

        #initialize "last deltas" as zeros
        self.lastInputToHiddenDeltas = np.zeros((1, self.num_inputs + 1))

        #hidden to output weights:
        self.hiddenToOutputWeights = np.random.uniform(-0.05, 0.05, (self.num_outputs, self.num_hidden + 1))
        self.lastHiddenToOutputDeltas = np.zeros((1, self.num_hidden + 1))
        
        self.data = np.c_[self.inputBiasValues, self.data]

        #confusion matrix setup: accuracy over time per perceptron
        self.confusionMatrix = np.zeros((10,2))

        sequence = np.arange(10).reshape((10,1))

        self.confusionMatrix = np.concatenate([sequence, self.confusionMatrix], 1)


        #history: accuracy over time (3 columns are epoch, total, and correct)
        #the percentage can be calculated afterward
        self.history = np.zeros((self.epochs, 3))


    #Reads in a new data set. Training data sets were split in two to avoid stack overflow
    #errors. Alternative solution could have been to use 16-bit floating point numbers
    #instead of 32-bit, but this works anyway.
    #Note: this function does NOT update weights. Can also be used for testing data.
    def updateInput(self, fileName):
        self.inputFile = open(fileName, encoding='utf-8-sig')
        with self.inputFile as file:
            self.inputMatrix = [[float(digit) for digit in line.split(',')] for line in file]
        self.data = np.asarray(self.inputMatrix)
        self.data[:,1:] = self.data[:,1:]/255
        self.inputBiasValues = np.full((len(self.data), 1), 1)
        self.data = np.c_[self.thresholds, self.data]


    #Analyze data set. Loops through all data entries/epochs and does the following:
        #1) computes dot product
        #2) determines whether dot product plus bias value is above 0 (yk)
        #3) makes prediction and compares with correct value
        #4) if "training" argument (a boolean) is TRUE then update weights
            #if not (testing phase) then do NOT update weights
        #5) compute accuracy for this epoch
    def analyze(self, training, orderIndex):
        num_entries = len(self.data)
        for e in range(self.epochs):
            print("Epoch #: ", e)
            correct_response = 0 #number of correct responses
            
            for n in range(len(self.data)):
                #keep track of the data in this row for debugging:
                self.this_row = self.data[n,:]
                yk = np.zeros(self.num_hidden)

                #compute dot product
                hiddenDotProduct = np.dot(self.data[n,], self.inputToHiddenWeights.transpose())

                #apply sigmoid function to hidden dot product
                self.hiddenActivationResult = np.apply_along_axis(self.sigmoid, 0, hiddenDotProduct)

                #compute dot product of hidden to outputs
                outputDotProduct = np.dot(self.hiddenActivationResult, self.hiddenToOutputWeights.transpose()[1:,:]) + self.hiddenToOutputWeights.transpose()[0,:]

                #apply sigmoid function to output dot product
                self.outputActivationResult = np.apply_along_axis(self.sigmoid, 0, outputDotProduct)

                #make prediction (highest output from activation result)
                predictions = np.argmax(self.outputActivationResult)

                #compare prediction with actual
                isequal = np.equal(predictions, self.data[n,1])

                #create "correctness" array represented by tk
                #0.1 for wrong answers and 0.9 for correct ones
                self.tk = np.zeros(self.num_outputs)
                self.tk += 0.1
                cval = self.data[n,1].astype(int)
                self.tk[cval] = 0.9

                self.backPropagate()

                #update accuracy
                num_correct = len(isequal[isequal > 0])
                correct_response += num_correct
                
                #do different things depending on whether this is test phase or training phase
                if(training == False):
                    #update confusion matrix:
                    self.confMatrix[cval, 2] += 1
                    self.confMatrix[cval, 1] += num_correct
                
            #calculate percent correct
            pc = 100 * correct_response/num_entries

            #print and record accuracy
            print("%: ", float("{0:.2f}".format(pc)))
            self.history[e, orderIndex] = pc

    #backpropagate
    def backPropagate(self):
        #calculate output and hidden errors
        self.outputError = self.outputActivationResult * (1 - self.outputActivationResult) * (self.tk - self.outputActivationResult)
        self.hiddenError = np.dot(self.outputError, self.hiddenToOutputWeights[:,1:]) * self.hiddenActivationResult * (1 - self.hiddenActivationResult)


        #calculate deltas and update old weights
        self.hiddenActivationResult = np.concatenate([[1], self.hiddenActivationResult])
        self.newHiddenToOutputDeltas = self.learning_rate * np.outer(self.outputError, self.hiddenActivationResult) + self.momentum * self.lastHiddenToOutputDeltas

        self.hiddenToOutputWeights += self.newHiddenToOutputDeltas
        self.lastHiddenToOutputDeltas = self.newHiddenToOutputDeltas        #keep track of last deltas for momentum

        self.newInputToHiddenDeltas = self.learning_rate * np.outer(self.hiddenError, self.this_row) + self.momentum * self.lastInputToHiddenDeltas
        self.inputToHiddenWeights += self.newInputToHiddenDeltas
        self.lastInputToHiddenDeltas = self.newInputToHiddenDeltas

    #print confusion matrix:
    def printConfusionMatrix(self):
        print(self.confusionMatrix)

    #print history (accuracy graphs) using matplotlib:
    def printHistory(self):
        plt.plot(self.history)
        plt.show()


#Execution begins here.
#classifier constructor: num_outputs, learning_rate, num_inputs, num_hidden, momentum, epochs
c = Classifier(10, 0.1, 785, 20, 0, 50)
c.read("data/mnist_train_part1.csv")
c.analyze(True, 0)
c.updateInput("data/mnist_train_part2.csv")
c.analyze(True, 1)
c.updateInput("data/mnist_test.csv")
c.analyze(False, 2)
c.printConfusionMatrix()
c.printHistory()

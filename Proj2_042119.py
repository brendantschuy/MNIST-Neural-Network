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
        try:
            ret = 1 / (1 + np.exp(val * -1))
            #print("no overflow on this one!")
        except:
            print("overflow error with val = ", val)
            #np.apply_along_axis(self.sigmoid, 0, val)
            ret = 1
        return ret


    #read in the input and organize it in data structures (matrices/numPy arrays)
    def read(self, fileName):
        self.inputFile = open(fileName)#, encoding='utf-8-sig')

        with self.inputFile as file:
            self.inputMatrix = [[float(digit) for digit in line.split(',')] for line in file]

        self.data = np.asarray(self.inputMatrix)
        self.data[:,1:] = self.data[:,1:]/255

        #not shuffling while debugging:
        #np.random.shuffle(self.data)

        #bias value is always 1:
        self.inputBiasValues = np.full((len(self.data), 1), 1)

        #randomize initial weights:
        #input to hidden weights:
        self.inputToHiddenWeights = np.random.uniform(-0.05, 0.05, (self.num_hidden, self.num_inputs + 1))
        #self.inputToHiddenWeights = np.zeros((self.num_hidden, self.num_inputs + 1))
        #self.inputToHiddenWeights += 0.05
        #print(self.inputToHiddenWeights)
        self.lastInputToHiddenDeltas = np.zeros((1, self.num_inputs + 1))
        #self.weights = np.c_[np.ones(self.num_hidden), self.randomWeights]

        #hidden to output weights:
        self.hiddenToOutputWeights = np.random.uniform(-0.05, 0.05, (self.num_outputs, self.num_hidden + 1))
        #self.hiddenToOutputWeights = np.zeros((self.num_outputs, self.num_hidden + 1))
        #self.hiddenToOutputWeights += 0.05
        self.lastHiddenToOutputDeltas = np.zeros((1, self.num_hidden + 1))
        
        self.data = np.c_[self.inputBiasValues, self.data]

        #confusion matrix: accuracy over time per perceptron
        self.confusionMatrix = np.zeros((10,2))
        #print(self.confusionMatrix)
        sequence = np.arange(10).reshape((10,1))
        #print(sequence)
        self.confusionMatrix = np.concatenate([sequence, self.confusionMatrix], 1)
        #print(self.confusionMatrix)

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
        #print("ALL DATA: ", self.data)
        for e in range(self.epochs):
            #print("------------------")
            print("Epoch #: ", e)
            correct_response = 0 #number of correct responses
            
            for n in range(len(self.data)):
                self.this_row = self.data[n,:]
                yk = np.zeros(self.num_hidden)

                #compute dot product
                #print("data row size: (should be 1x785)", self.data[n,].shape)
                #print("input to hidden weights shape: ", self.inputToHiddenWeights.shape)
                #print(f"before handling data from row {n}, I->H weights:", self.inputToHiddenWeights.transpose())
                #print(f"before handling data from row {n}, H->O weights:", self.hiddenToOutputWeights.transpose())
                hiddenDotProduct = np.dot(self.data[n,], self.inputToHiddenWeights.transpose())
                #print("hidden dot product shape: ", hiddenDotProduct.shape)

                #apply sigmoid function to hidden dot product
                self.hiddenActivationResult = np.apply_along_axis(self.sigmoid, 0, hiddenDotProduct)
                #print("hidden dot product: ", hiddenDotProduct)
                #print("hidden activation result shape: ", self.hiddenActivationResult.shape)

                #print("hidden to output weights shape: ", self.hiddenToOutputWeights.shape)

                #compute dot product of hidden to outputs
                outputDotProduct = np.dot(self.hiddenActivationResult, self.hiddenToOutputWeights.transpose()[1:,:]) + self.hiddenToOutputWeights.transpose()[0,:]

                #print("output dot product shape: ", outputDotProduct.shape)

                #apply sigmoid function to output dot product
                self.outputActivationResult = np.apply_along_axis(self.sigmoid, 0, outputDotProduct)
                #print("activation result shape: (should be 10x1 for each data point", self.outputActivationResult.shape)

                #make prediction (highest output from activation result)
                predictions = np.argmax(self.outputActivationResult)

                #compare prediction with actual
                #print("predictions: ", predictions)
                #print("actual value: ", self.data[n,1])
                isequal = np.equal(predictions, self.data[n,1])

                #create "correctness" array represented by tk
                #0.1 for wrong answers and 0.9 for correct ones
                self.tk = np.zeros(self.num_outputs)
                self.tk += 0.1
                #print("result: ", self.data[:,1])
                cval = self.data[n,1].astype(int)
                #print(self.data)
                #print("cval = self.data[n,1] = ", cval)
                #print("input data: ", self.data[:,1])

                #for the sake of small-scale debugging: > 0.5 -> 0.9
                #self.tk[self.tk > 0.5] = 0.9
                self.tk[cval] = 0.9

                #print(self.tk)

                #print("DATA THIS ROW: ", self.data[n,])
                self.backPropagate()





                #stop here

                #calculate difference (tk - yk)
                #diff = tk - yk

                #total up number of correct responses
                num_correct = len(isequal[isequal > 0])
                correct_response += num_correct

                #update weights
                #self.inputToHiddenWeights += LR * np.outer(diff, self.data[n,])
                #self.inputToHiddenWeights.transpose()[0] += self.learning_rate * diff
                
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
        #requires self.outputActivationResult
        #print("output activation result shape (should be 10x1): ", self.outputActivationResult.shape)
        #print(self.outputActivationResult)
        #print("correctness activation result shape (should be 10x1): ", self.tk.shape)
        #print(self.tk)
        self.outputError = self.outputActivationResult * (1 - self.outputActivationResult) * (self.tk - self.outputActivationResult)
        #print("output error: ", self.outputError)
        #print("output error shape: (should be 10x1): ", self.outputError.shape)
        #print(self.outputActivationResult)
        #print(self.outputError)
        #print(self.tk)
        self.hiddenError = np.dot(self.outputError, self.hiddenToOutputWeights[:,1:]) * self.hiddenActivationResult * (1 - self.hiddenActivationResult)
        #print(self.hiddenToOutputWeights.shape)
        #print(self.outputActivationResult.shape)
        #print(self.outputError.shape)
        #hiddenToOutputWeights: 10, 21
        #print(self.hiddenToOutputWeights.shape)
        #print("before: ", self.hiddenToOutputWeights)
        #print("HIDDEN ACT RESULT: ", self.hiddenActivationResult)
        #print("OUT ACT RESULT: ", self. outputActivationResult)
        #print("TRUTH VALUES: ", self.tk)
        #print("self.outputError: ", self.outputError)
        #print("self.hiddenError = ", self.hiddenError)
        #print("self.lastHiddenToOutput: ", self.lastHiddenToOutputDeltas)
        #print("error dot activation: ", np.dot(self.outputError, self.hiddenToOutputWeights[:,1:]))
        

        #print("hidden activation values: ", self.hiddenActivationResult)
        self.hiddenActivationResult = np.concatenate([[1], self.hiddenActivationResult])
        #print("now hidden activation values: ", self.hiddenActivationResult)
        #hiddenBackProp = np.insert(self.hiddenActivationResult, 0, 1, axis=0)
        #print("now hidden Activation Result:", hiddenBackProp)
        #print("weights shape before changing: ", self.hiddenToOutputWeights.shape)
        self.newHiddenToOutputDeltas = self.learning_rate * np.outer(self.outputError, self.hiddenActivationResult) + self.momentum * self.lastHiddenToOutputDeltas
        #print("new hidden to output deltas: ", self.newHiddenToOutputDeltas)
        self.hiddenToOutputWeights += self.newHiddenToOutputDeltas
        self.lastHiddenToOutputDeltas = self.newHiddenToOutputDeltas
        #print("after: ", self.hiddenToOutputWeights)
        self.newInputToHiddenDeltas = self.learning_rate * np.outer(self.hiddenError, self.this_row) + self.momentum * self.lastInputToHiddenDeltas
        self.inputToHiddenWeights += self.newInputToHiddenDeltas
        self.lastInputToHiddenDeltas = self.newInputToHiddenDeltas

        #print("Last I->H weight deltas now = ", self.lastInputToHiddenDeltas)
        #print("Last H->O weight deltas now = ", self.lastHiddenToOutputDeltas)

        #print("I->H WEIGHTS: ", self.inputToHiddenWeights)
        #print("H->O WEIGHTS: ", self.hiddenToOutputWeights)

    #print confusion matrix:
    def printConfusionMatrix(self):
        #print(f"after end, I->H weights:", self.inputToHiddenWeights.transpose())
        #print(f"after end, H->O weights:", self.hiddenToOutputWeights.transpose())
        print(self.confusionMatrix)

    #print history (accuracy graphs) using matplotlib:
    def printHistory(self):
        plt.plot(self.history)
        plt.show()


#Execution begins here.
#classifier constructor: num_outputs, learning_rate, num_inputs, num_hidden, momentum, epochs
c = Classifier(10, 0.1, 785, 100, 0, 50)
#c = Classifier(2, 0.1, 3, 2, 0, 1)
#c.read("data/microset.csv")
c.read("data/mnist_train.csv")
c.analyze(True, 0)
#c.updateInput("data/mnist_train_part2.csv")
#c.analyze(True, 1)
c.updateInput("data/mnist_test.csv")
c.analyze(False, 2)
c.printConfusionMatrix()
c.printHistory()

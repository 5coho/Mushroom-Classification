"""
--MushroomDataHandler--

This class takes the data from the files:
    -MushroomData_8000.txt
    -MushroomData_Unknown_100.txt

Encodes them, coverts to numpy array and splits them so they are usable
by sklearn.

"""


#Metadata
__author__          = "Scott Howes"
__credits__         = "Scott Howes"
__email__           = "showes@unbc.ca"
__python_version__  = "3.9.4"


#imports
from sklearn.preprocessing import LabelEncoder
import numpy


# class to solve the knapsack 0-1 problem
class MushroomDataHandler:


    def __init__(self):
        pass


    #getting the data from file, parses, converts to numpy array then Encodes
    #returns trainingData, trainingLabels
    def getTrainingData(self, filePath):

        #opening the file
        dataFile = open(filePath, "r")

        #making a list with the file
        tempList = dataFile.readlines()

        #closing the file
        dataFile.close()

        #splitting the data
        itemList = []
        for items in tempList:
            itemList.append(items.split())

        #splitting  the data again
        splitList = []
        for items in itemList:
            for attributes in items:
                splitList.append(attributes.split(","))

        #creating the class list by popping first element
        classList = []
        for items in splitList:
            classList.append(items.pop(0)[0])

        #converting to numpy arrays
        trainingData = numpy.array(splitList)
        trainingLabels = numpy.array(classList)

        #changing array values from chars to ints
        encoder = LabelEncoder()

        for col in range(len(trainingData[0])):
            trainingData[:, col] = encoder.fit_transform(trainingData[:, col])

        trainingLabels = encoder.fit_transform(trainingLabels)

        #changing data type of trainingData to float
        trainingData = trainingData.astype(float)

        return trainingData, trainingLabels


    #this function gets the unknown data and returns a numpy array
    def getUnknownData(self, filePath):

        #opening the file
        dataFile = open(filePath, "r")

        #making a list with the file
        tempList = dataFile.readlines()

        #closing the file
        dataFile.close()

        #splitting the data
        itemList = []
        for items in tempList:
            itemList.append(items.split())

        #splitting  the data again
        splitList = []
        for items in itemList:
            for attributes in items:
                splitList.append(attributes.split(","))


        #converting to numpy arrays
        trainingData = numpy.array(splitList)

        #changing array values from chars to ints
        encoder = LabelEncoder()

        for col in range(len(trainingData[0])):
            trainingData[:, col] = encoder.fit_transform(trainingData[:, col])

        #changing data type of trainingData to float
        trainingData = trainingData.astype(float)

        return trainingData


    #this function prints the answers to file
    def printPredictionsToFile(self, predictions, filePath):

        #e and p map
        classMap = {0:"e", 1:"p"}

        #converint numpy array to list
        predictions.tolist()

        #creating file
        outputFile = open(filePath, "w")

        for prediction in predictions:
            outputFile.write(f"{classMap[prediction]}\n")

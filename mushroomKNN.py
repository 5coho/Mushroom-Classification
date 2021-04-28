"""
--K Nearest Neighbor for Mushroom Classification--


Uses sklearn to classify if a mushroom is edible or poisonous

"""


#Metadata
__author__          = "Scott Howes"
__credits__         = "Scott Howes"
__email__           = "showes@unbc.ca"
__python_version__  = "3.9.4"


#imports
from MushroomDataHandler import MushroomDataHandler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import neighbors


if __name__ == "__main__":

    #variables
    k = 5
    trainingFile = "MushroomData_8000.txt"
    unknownFile = "MushroomData_Unknown_100.txt"
    printFile = "predictionResultsKNN.txt"

    #getting the data for classification
    dataHandler = MushroomDataHandler()

    print()
    print("Starting KNN Mushroom Classification", flush=True)
    print()
    print("Loading Data...", flush=True)
    print()

    X, y = dataHandler.getTrainingData(trainingFile)

    print("Splitting Data 80-20...", flush=True)
    print()

    #splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    print(f"X_train size:  {X_train.shape[0]}", flush=True)
    print(f"y_train size:  {y_train.shape[0]}", flush=True)
    print(f"X_test size:   {X_test.shape[0]}", flush=True)
    print(f"y_test size:   {y_test.shape[0]}", flush=True)
    print()

    #this loop checks which value for k is best
    #k=5 seems to be a good bet
    #for k in range(1,51):
    #    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    #    knn.fit(X_train, y_train)
    #    print(knn.score(X_test, y_test), flush=True)

    print("Training K Nearest Neighbor Classifier...", flush=True)

    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    print("DONE!", flush=True)
    print()
    print(f"Accurany Score: {knn.score(X_test, y_test)}", flush=True)
    print()

    #testPredictions = knn.predict(X_test)
    #print(classification_report(y_test, testPredictions))

    print("Loading Unknown data...", flush=True)

    unknowns = dataHandler.getUnknownData(unknownFile)

    print("Predicting...", flush=True)

    predictions = knn.predict(unknowns)

    print("DONE!", flush=True)
    print()

    print("Printing predictions to file...", flush=True)

    dataHandler.printPredictionsToFile(predictions, printFile)

    print("DONE!", flush=True)
    print()

    print(f"File saved as {printFile}", flush=True)
    print()

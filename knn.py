import numpy as np
import pandas as pd
from operator import itemgetter

dataset1 = pd.read_csv('yeast_training (1).csv', header=None)
dataset2 = pd.read_csv('yeast_test2.csv', header=None)

train_dataset = dataset1.values[:, 0:8]
test_dataset = dataset2.values[:, 0:8]

outputTrain = (dataset1.values[0:, [8]])
outputTest = (dataset2.values[0:, 8])

x1 = train_dataset.min(axis=0)
x2 = train_dataset.max(axis=0)
train_dataset = (train_dataset - x1) / (x2-x1)
test_dataset = (test_dataset-x1)/(x2-x1)


def euclideanDistance(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    result = np.linalg.norm(v1 - v2)
    return result


def NearestNeighbours(rowOfTest, k):
    allDistances = []
    for j in range(len(train_dataset)):
        result = euclideanDistance(rowOfTest, train_dataset[j])
        allDistances.append((result, outputTrain[j]))
    allDistances = sorted(allDistances, key=itemgetter(0))
    neighbours = getNeighbours(k, allDistances)
    return neighbours


def getNeighbours(k, allDistances):
    nearestNeighbours = []
    for i in range(k):
        nearestNeighbours.append(allDistances[i])
    return nearestNeighbours


def KNN_Classification(rowOfTest, k):
    #check if k is even
    if k % 2 == 0:
        k = k + 1
    
    outputs = []
    listNeighbours = NearestNeighbours(rowOfTest, k)
    listNeighbours = np.array(listNeighbours, dtype="object")
    for i in range(len(listNeighbours)):
        outputs.append(listNeighbours[i, 1])
    predicted = majority(outputs)
    return predicted


def majority(List):
    counter = 0
    string = List[0]
    for i in List:
        indexCount = List.count(i)
        if (indexCount > counter):
            string = i
            counter = indexCount

    return string


def sendTestRows(k):
    count = 0
    for i in range(len(test_dataset)):
        predicted = KNN_Classification(test_dataset[i], k)
        if predicted == outputTest[i]:
            count = count + 1
    return count


def getAccuracy(k):
    count = sendTestRows(k)
    acc = count / len(test_dataset) * 100
    newacc = format(acc, '.2f')
    #print all expected and predicted values
    for i in range(len(test_dataset)):
        predicted = KNN_Classification(test_dataset[i], k)
        print("Expected: " + str(outputTest[i]) + " Predicted: " + str(predicted))
    print("Number of correctly classified instances:", count, ",Total number of instances:", len(outputTest),
          ", Accuracy:", newacc, "%")


k = 1
print("At K=1")
print("predicted class : " +
      str(KNN_Classification(test_dataset[k], k)) + " Actual class: " + str(outputTest[k-1]))
getAccuracy(k)

k = 2
print("At K=2")
print("predicted class : " +
      str(KNN_Classification(test_dataset[k], k)) + " Actual class: " + str(outputTest[k-1]))
getAccuracy(k)

k = 3
print("At K=3")
print("predicted class : " +
      str(KNN_Classification(test_dataset[k], k)) + " Actual class: " + str(outputTest[k-1]))
getAccuracy(k)

k = 4
print("At K=4")
print("predicted class : " +
      str(KNN_Classification(test_dataset[k], k)) + " Actual class: " + str(outputTest[k-1]))
getAccuracy(k)

k = 5
print("At K=5")
print("predicted class : " +
      str(KNN_Classification(test_dataset[k], k)) + " Actual class: " + str(outputTest[k-1]))
getAccuracy(k)

k = 6
print("At K=6")
print("predicted class : " +
      str(KNN_Classification(test_dataset[k], k)) + " Actual class: " + str(outputTest[k-1]))
getAccuracy(k)

k = 7
print("At K=7")
print("predicted class : " +
      str(KNN_Classification(test_dataset[k], k)) + " Actual class: " + str(outputTest[k-1]))
getAccuracy(k)

k = 8
print("At K=8")
print("predicted class : " +
      str(KNN_Classification(test_dataset[k], k)) + " Actual class: " + str(outputTest[k-1]))
getAccuracy(k)

k = 9
print("At K=9")
print("predicted class : " +
      str(KNN_Classification(test_dataset[k], k)) + " Actual class: " + str(outputTest[k-1]))
getAccuracy(k)

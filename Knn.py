#!/usr/bin/python3
# -*-coding:utf-8-*-
from numpy import *
import operator
import csv

#str转int
def toInt(array):
    array = mat(array)
    m, n = shape(array)
    newArray = zeros((m, n))
    for i in range(m):
        for j in range(n):
            newArray[i, j] = int(array[i, j])
    return newArray

#序列化
def nomalizing(array):
    m, n = shape(array)
    for i in range(m):
        for j in range(n):
            if array[i, j] != 0:
                array[i, j] = 1
    return array


def loadTrainData():
    l = []
    with open('train.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)  # 42001*785
    l.remove(l[0])
    l = array(l)
    label = l[:, 0]
    data = l[:, 1:]
    return nomalizing(toInt(data)), toInt(label)  # label 1*42000  data 42000*253
    # return data,label


def loadTestData():
    l = []
    with open('test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
            # 28001*784
    l.remove(l[0])
    data = array(l)
    return nomalizing(toInt(data))  # data 28000*253



# dataSet:m*n   labels:m*1  inX:1*n

#knn分类
def classify(inX, dataSet, labels, k):
    inX = mat(inX)
    dataSet = mat(dataSet)
    labels = mat(labels)
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = array(diffMat) ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i], 0]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]  #返回最大value的key


def saveResult(result):
    with open('result.csv', 'w') as myFile:
        myWriter = csv.writer(myFile)
        myWriter.writerow(["ImageId","Label"])  #添加属性名与序
        index = 0
        for i in result:
            tmp = []
            index = index + 1
            tmp.append(index)
            tmp.append(int(i))
            myWriter.writerow(tmp)#将列表以行写入


def handwritingClassTest():

    print('正常运行')
    trainData, trainLabel = loadTrainData()
    testData = loadTestData()
   # testLabel = loadTestResult()
    m, n = shape(testData)
    errorCount = 0
    resultList = []
    for i in range(m):
        classifierResult = classify(testData[i], trainData, trainLabel.transpose(), 5)
        resultList.append(classifierResult)
        print(i)
    saveResult(resultList)


handwritingClassTest()



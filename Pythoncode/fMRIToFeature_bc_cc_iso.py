# Author Dr.Mei-Chih Chang
# Chair of Information Architecture ETH Zürich
# Taiwanese Workshop : Big-Data Informed Urban Design for Smart City : Workshop 17-19.Nov.2017

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from itertools import cycle

def listToMat(l):
    return np.array(l).reshape(-1, np.array(l).shape[-1])

def plotter(measured, predicted):
    # plot the result
    fig, ax = plt.subplots()
    y = measured
    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
    return 0

def plot_2D(data, target, target_names, filename):
    colors = cycle('rgbcmykw')
    target_ids = range(len(target_names))
    plt.figure()
    ax = plt.gca()
    ax.set_facecolor('white')
    for i, c, label in zip(target_ids, colors, target_names):
        plt.scatter(data[target == i, 0], data[target == i, 1], c=c, label=label)
    plt.legend()
    plt.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0)
    return 0

def readXlsxFileDir(fileDir):
    dirFile = []
    for file in os.listdir(fileDir):
        if file.endswith(".xlsx"):
            dirFile.append(file)
    dirFile = listToMat(dirFile)
    numDir = dirFile.shape[1]
    if numDir == 0:
        return [], -1
    else:
        return dirFile, numDir

def readCsvFileDir(fileDir):
    dirFile = []
    for file in os.listdir(fileDir):
        if file.endswith(".csv"):
            dirFile.append(file)
    dirFile = listToMat(dirFile)
    numDir = dirFile.shape[1]
    if numDir == 0:
        return [], -1
    else:
        return dirFile, numDir

def readFileNumber(filename):
    startIndex = filename.find('_')
    endIndex = filename.find('.')
    if (startIndex != -1) & (endIndex > startIndex):
        return int(filename[startIndex + 1:endIndex])
    else:
        return -1

def computeStats(yy):
    temp = []
    temp.append(yy.mean())
    temp.append(yy.var())
    temp.append(yy.std())
    temp.append(np.median(yy))
    temp.append(stats.skew(yy))
    temp.append(stats.kurtosis(yy))
    temp.append(stats.moment(yy, 3))
    temp.append(stats.moment(yy, 4))
    return (temp)

def mriToStatsFeature(fileDir1: object, fileDir2: object, fileDir3: object) -> object:
    import pandas as pd
    file1, number1 = readXlsxFileDir(fileDir1)
    file2, number2 = readXlsxFileDir(fileDir2)
    file3, number3 = readXlsxFileDir(fileDir3)
    numberIsovist = 4
    numberStats = 8
    features = np.zeros([number1, numberIsovist+(numberStats)*2])
    filetable = []

    # get data
    for i in range(number1):
        filename1 = file1[0, i]
        filename2 = file2[0, i]
        filename3 = file3[0, i]

        filetable.append(readFileNumber(fileDir1 + filename1))

        # for bc
        x1 = pd.read_excel(fileDir1 + filename1, sheetname=0, header=0, dtype=np.float64)
        data1 = x1.values.flatten()
        for index in range(len(data1)):
            if (np.isinf(data1[index])):
                data1[index] = 0
                print(i, index)
        features[i, :numberStats] = computeStats(data1)

        # for cc
        x2 = pd.read_excel(fileDir2 + filename2, sheetname=0, header=0, dtype=np.float64)
        data2 = x2.values.flatten()
        for index in range(len(data2)):
            if (np.isinf(data2[index])):
                data2[index] = 0
                print(i, index)
        features[i, numberStats:numberStats * 2] = computeStats(data2)

        # for iso
        x3 = pd.read_excel(fileDir3 + filename3, sheetname=0, header=0, dtype=np.float64)
        data3 = x3.values.flatten()
        for index in range(len(data3)):
            if (np.isinf(data3[index])):
                data3[index] = 0
                print(i, index)
        features[i, numberStats * 2:] = data3

    return features, filetable




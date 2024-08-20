#!/usr/bin/env python
# coding: utf-8

# In[2]:


# # Convert to python script, remember to delete/comment the next line in the actual file
# ! jupyter nbconvert --to python twoClassClassification.ipynb --output test14-8.py

# # Run the notebook in Simpson GPU server
# CUDA_VISIBLE_DEVICES=0 python testSamples2-8.py -batchSize=16 -epochs=100 -lr=0.001 -evalDetailLine="majourity voting on smote with 2 clases" -hasBackground=f -usesLargestBox=f -segmentsMultiple=12 -dropoutRate=0.2 -grouped2D=t -modelChosen='Small2D' -votingSystem='majority'
# CUDA_VISIBLE_DEVICES=1 python test14-8.py && CUDA_VISIBLE_DEVICES=1 python test14-8.py

## Instantiate the values of the model
# python testSamples2-8.py -batchSize=8 -epochs=100 -lr=0.001 -evalDetailLine="majourity voting on new data" -hasBackground=f -usesLargestBox=f -segmentsMultiple=12 -dropoutRate=0.2 -grouped2D=t -modelChosen='Small2D' -votingSystem='majority'




# ### # Imports

# In[25]:


# Image reading and file handling 
import pandas as pd
import SimpleITK as sitk 
import os 
import shutil
from collections import Counter


# Image agumentaitons 
import numpy as np
import cv2
from PIL import Image
import random

# Saving History
import pickle as pkl

# Train test set spliting
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold

# Dataset building
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import random
from sklearn.model_selection import StratifiedKFold

# Model building
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torch.optim import lr_scheduler
import timm # For Xception model

# Evaluation metrics and Plotting
import matplotlib.pyplot as plt
import scipy.stats
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Oversampling
from imblearn.over_sampling import SMOTE


# Import the methods from twoClassClassificationMethods
# import sys, importlib
# importlib.reload(sys.modules['ipynb.fs.full.twoClassClassificaitonMethods'])
# from ipynb.fs.full.twoClassClassificaitonMethods import *
# importlib.reload(sys.modules['twoClassClassificaitonMethods'])
from twoClassClassificationMethods import *


# In[26]:


# ! pip freeze > requirements.txt
# ! pip uninstall -y -r requirements.txt

## Make a python environment
# ! python3.8 -m venv threeDresearchPip

## Download necessary packages 
# ! pip install matplotlib opencv-python scipy simpleitk pandas openpyxl scikit-learn nbconvert imblearn
# ! pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 

## May need to download networkx 3.1 because of older python version of torch
# ! pip install networkx==3.1

## For 3D image classification
# ! pip install foundation-cancer-image-biomarker -qq
# ! pip install foundation-cancer-image-biomarker
# ! pip3 install torchio

## In case pip breaks 
# ! python -m ensurepip --upgrade

## Check python version and packages
# ! python --version
# ! pip3 freeze > research3D.txt


# In[38]:


def generateKFoldsValidation(testInformation, dataset, dataframe, k=2, trainingTransform=None):
    
#     testInformation = {
#     'testName' : testName,
#     'batchSize': batchSize,
#     'numOfEpochs': numOfEpochs,
#     'evalDetailLine': evalDetailLine,
#     'learningRate': learningRate,
#     'hasBackground': hasBackground,
#     'usesLargestBox': usesLargestBox,
#     'segmentsMultiple': segmentsMultiple,
#     'grouped2D': grouped2D,
#     'weight_decay': weight_decay,
#     'modelChosen': modelChosen,
#     'votingSystem': votingSystem,
#     'patience': patience,
#     'sampleStrategy': sampleStrategy
# }
    randomSeed = 42
    seed_everything(randomSeed)

    patients = list(dataset.keys())
    fakeData = [-1] * len(patients)
    stratifiedFolds = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    stratifiedFolds.get_n_splits(patients, fakeData)
    splits = enumerate(stratifiedFolds.split(patients,fakeData))


    accuracies = []
    f1s = []
    recalls = []
    predictionSplits = []
    precisions = []
    rocAucs = []
    endingEpochs = []

    histories = []
    confusion_matricies = []
    rocCurves = []

    print(f"\n\n====================Begin testing for {testInformation['evalDetailLine']}====================")

    for i, (trainIndicies, testIndicies) in splits:
        patients = list(dataset.keys())
        trainFolders = [patients[i] for i in trainIndicies]
        testFolders = [patients[i] for i in testIndicies]
        valFolders = testFolders

        trainData, valData, testData, training_data_transforms = convertDataToLoaders(trainFolders, valFolders, testFolders, dataset, 
                                                                                      testInformation['modelChosen'], testInformation['grouped2D'], 
                                                                                      testInformation['segmentsMultiple'], training_data_transforms = None, 
                                                                                      batchSize=testInformation['batchSize'])
        

        print(f"\n--------------------------------{testInformation['evalDetailLine']} -- Fold #{i+1}--------------------------------")
        print('Train Data:', len(trainData))
        print('Validation Data:', len(valData))
        print('Test Data:', len(testData))


        resultName = 'Tests/'+testInformation['testName']+'/'+testInformation['evalDetailLine']
        resultNameWithFold = resultName+f'/fold-{i+1}/'

        ## Select and Train Model
        model, criterion, scheduler, optimizer = defineModel(learningRate=testInformation['learningRate'], weight_decay=testInformation['weight_decay'], 
                                                             model = testInformation['modelChosen'])
        model, criterion, device, history, endingEpoch = trainModel(model, criterion, scheduler, optimizer, trainData, valData, 
                                                                    patience=testInformation['patience'],numOfEpochs=testInformation['numOfEpochs'])

        saveResults(resultNameWithFold, model, history, training_data_transforms, saveModel=False)

        #Evaluate perforamnce on test set
        confusionMatrixDisp, rocCurveDisplay, testingMetrics = evaluateModelOnTestSet(resultNameWithFold, model, testData, criterion, device, 
                                                                                      testInformation['votingSystem'], testInformation['segmentsMultiple'], 
                                                                                      saveConfusionMatrix = False, showConfusionMatrix=False,
                                                                                      showROCCurve=False, saveROCCurve=False)
        
        plotTraining(resultNameWithFold, '-', history, saveFigure=False, showResult=False)

        # Collect metrics
        accuracies.append(testingMetrics['Accuracy'])
        f1s.append(testingMetrics['F1'])
        recalls.append(testingMetrics['Recall'])
        predictionSplits.append(testingMetrics['PredictionSplits'])
        precisions.append(testingMetrics['Precision'])
        rocAucs.append(testingMetrics['ROC-AUC'])
        endingEpochs.append(endingEpoch)
        
        histories.append(history)
        confusion_matricies.append(confusionMatrixDisp)
        rocCurves.append(rocCurveDisplay)

    kFoldsTestMetrics = {'Accuracy':meanConfidenceInterval(accuracies), 'F1':averageMultilabelMetricScores(f1s), 'Recall':averageMultilabelMetricScores(recalls), 
                    'PredictionSplits':averagePredictionTotals(predictionSplits), 'Precision':averageMultilabelMetricScores(precisions), 
                    'ROC-AUC':meanConfidenceInterval(rocAucs), 'endingEpochs':endingEpochs}
    
    # Write the test information and testvalues to files
    print(f"\n--------------------------------{testInformation['evalDetailLine']} -- AVERAGES --------------------------------")
    writeDictionaryToTxtFile(resultName+'/kFoldsTestMetrics.txt',kFoldsTestMetrics, printLine=True)
    writeDictionaryToTxtFile(resultName+'/testInformation.txt',testInformation, printLine=False)
    
    print('\n\n')
    # Plot training, confusion matrix, and roc curves for each fold as a single .png
    plotConfusionMatricies(resultName, f"{testInformation['evalDetailLine']}", confusion_matricies)
    plotROCCurves(resultName, f"{testInformation['evalDetailLine']}", rocCurves)
    plotTrainingPerformances(resultName, f"{testInformation['evalDetailLine']}", histories, saveFigure=True, showResult=True)

    appendMetricsToXLSX(testInformation['evalDetailLine'], testInformation['testName'], kFoldsTestMetrics, dataframe)

    #Make copies of the two scripts
    for filename in os.listdir():
        # Check if the file ends with .py
        if filename.endswith('.py'):
            # Copy the .py file
            shutil.copy(filename, resultName+'/'+filename)


# In[ ]:


def loadFromPickle(name):
    with open(f'{name}.pkl', 'rb') as fp:
        data = pkl.load(fp)
    return data  

def getDatasetShape(data):
    imageSize = data[list(data.keys())[0]]['images'].shape
    return [len(data), imageSize[0],imageSize[1],imageSize[2]]

def checkShapesConsistent(data):
    keys = list(data.keys())
    size = data[keys[0]]['images'].shape
    for i in range(len(data)):    
        if size != data[keys[i]]['images'].shape:
            print(f"Error in shape at index {i} with shape {data[keys[i]]['images'].shape}")
            return False , size
    return True , size
    
def getDataset(testInformation):
    # Set random seeds
    randomSeed = 42
    seed_everything(randomSeed)

    ## LOAD THE DATA
    ## ==============================================================================================================
    name = f"preprocessCombinations/hasBackground={testInformation['hasBackground']}-usesLargestBox={testInformation['usesLargestBox']}-segmentsMultiple={testInformation['segmentsMultiple']}-size=(119,119)"

    dataset = loadFromPickle(name)
    consitencyCheck, instanceSize = checkShapesConsistent(dataset)
    print('Sizes are all the same? ', consitencyCheck)
    assert consitencyCheck
    print(f'dataset shape:')
    print(len(dataset), instanceSize)


    ## SPLIT THE DATA
    if testInformation['sampleStrategy'] == 'underSampling':
        dataset = underSampleData(dataset)
        
    # if sampleStrategy == 'overSampling':
    #     trainData = overSampleData(trainFolders)

    return dataset


# In[ ]:


columns = ['TestName','RunData','PredictionSplits','Accuracy','F1Average','RecallAverage','PrecisionAverage','ROC-AUC','EndingEpoch','AccuracyData','F1Data','RecallData','PrecisionData','ROC-AUCData']
dataframePath='testResultsNew.xlsx'
sheetName = 'KFolds'

def gridSearch(testInformation):
    for key, value in testInformation.items():
        print(f'{key}: {value}')

    dataframe = pd.read_excel(dataframePath, sheetName,header=None, names=columns)
    #addEvalDetailToModel(testInformation['evalDetailLine'],dataframe)

    dataset = getDataset(testInformation)
    generateKFoldsValidation(testInformation, dataset, dataframe, k=5, trainingTransform=None) 
    dataframe.to_excel(dataframePath, sheetName, index=False, header=False)

def setGridSearchParams():

    # Data Parameters
    testName = 'testSamples14-8/gridSearchInception'
    evalDetailLine = "test if k folds works"
    # batchSize = 16
    numOfEpochs = 50
    # learningRate = 0.001
    hasBackground = True
    usesLargestBox = True
    segmentsMultiple = 1
    grouped2D = False
    weight_decay = 0.01
    modelChosen = 'InceptionV3Small2D' #'ResNet50Small2D', 'VGG16Small2D', 'InceptionV3Small2D'
    votingSystem = 'singleLargest' #average, singleLargest
    patience = 10
    sampleStrategy = 'underSampling' # 'underSampling', 'overSampling', 'normal' 

    for modelChosen in ['XceptionSmall2D']:#,'InceptionV3Small2D','VGG16Small2D','XceptionSmall2D']: #, 'ResNet50Small2D','InceptionV3Small2D','VGG16Small2D', 'XceptionSmall2D']:
        for learningRate in [0.001,0.0001]:
            for weight_decay in [0.01,0.001]:
                for batchSize in [16,32]:
                    for patience in [5,10]:
                        testName = f'testSamples14-8/gridSearch{modelChosen}-segmentsMultiple=1'
                        evalDetailLine = f"-modelChosen={modelChosen}-segmentsMultiple={segmentsMultiple}-lr={learningRate}-weight_decay={weight_decay}-batchSize={batchSize}-patience={patience}"
                        testInformation = {
                            'testName' : testName,
                            'evalDetailLine' : evalDetailLine,
                            'batchSize': batchSize,
                            'numOfEpochs': numOfEpochs,
                            'learningRate': learningRate,
                            'hasBackground': hasBackground,
                            'usesLargestBox': usesLargestBox,
                            'segmentsMultiple': segmentsMultiple,
                            'grouped2D': grouped2D,
                            'weight_decay': weight_decay,
                            'modelChosen': modelChosen,
                            'votingSystem': votingSystem,
                            'patience': patience,
                            'sampleStrategy': sampleStrategy
                        }
                        gridSearch(testInformation)

## Reproduce the best model  
##-modelChosen=ResNet50Small2D-lr=0.001-weight_decay=0.01-batchSize=32-patience=5
# testName = 'testSamples14-8/gridSearchResnet'
# evalDetailLine = "see if segmentsMultiple=1 works with average"
# batchSize = 32
# numOfEpochs = 50
# learningRate = 0.001
# hasBackground = True
# usesLargestBox = True
# segmentsMultiple = 6
# grouped2D = True
# weight_decay = 0.01
# modelChosen = 'ResNet50Small2D' #'ResNet50Small2D', 'VGG16Small2D', 'InceptionV3Small2D'
# votingSystem = 'average' #average, singleLargest
# patience = 5
# sampleStrategy = 'underSampling' # 'underSampling', 'overSampling', 'normal' 

# testInformation = {
# 'testName' : testName,
# 'evalDetailLine' : evalDetailLine,
# 'batchSize': batchSize,
# 'numOfEpochs': numOfEpochs,
# 'learningRate': learningRate,
# 'hasBackground': hasBackground,
# 'usesLargestBox': usesLargestBox,
# 'segmentsMultiple': segmentsMultiple,
# 'grouped2D': grouped2D,
# 'weight_decay': weight_decay,
# 'modelChosen': modelChosen,
# 'votingSystem': votingSystem,
# 'patience': patience,
# 'sampleStrategy': sampleStrategy
# }
# gridSearch(testInformation)


setGridSearchParams()



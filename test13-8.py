#!/usr/bin/env python
# coding: utf-8

# In[42]:


# # Convert to python script, remember to delete/comment the next line in the actual file
# ! jupyter nbconvert --to python twoClassClassification.ipynb --output test13-8.py

# # Run the notebook in Simpson GPU server
# CUDA_VISIBLE_DEVICES=0 python testSamples2-8.py -batchSize=16 -epochs=100 -lr=0.001 -evalDetailLine="majourity voting on smote with 2 clases" -hasBackground=f -usesLargestBox=f -segmentsMultiple=12 -dropoutRate=0.2 -grouped2D=t -modelChosen='Small2D' -votingSystem='majority'
# CUDA_VISIBLE_DEVICES=0 python test13-8.py


# In[ ]:


# Data Parameters
testName = 'testSamples13-8'
evalDetailLine = "test if k folds works"
batchSize = 16
numOfEpochs = 2
learningRate = 0.001
hasBackground = True
usesLargestBox = True
segmentsMultiple = 6
grouped2D = True
weight_decay = 0.01
modelChosen = 'ResNet50Small2D' #'ResNet50Small2D', 'VGG16Small2D', 'InceptionV3Small2D'
votingSystem = 'average' #average, singleLargest
patience = 10
sampleStrategy = 'underSampling' # 'underSampling', 'overSampling', 'normal' 

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

for key, value in testInformation.items():
    print(f'{key}: {value}')


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


# ### Define Model and training

# In[27]:


# import sys, importlib
# importlib.reload(sys.modules['ipynb.fs.full.twoClassClassificaitonMethods'])
# from ipynb.fs.full.twoClassClassificaitonMethods import *
# importlib.reload(sys.modules['twoClassClassificaitonMethods'])
from twoClassClassificationMethods import *

## Instantiate the values of the model
# python testSamples2-8.py -batchSize=8 -epochs=100 -lr=0.001 -evalDetailLine="majourity voting on new data" -hasBackground=f -usesLargestBox=f -segmentsMultiple=12 -dropoutRate=0.2 -grouped2D=t -modelChosen='Small2D' -votingSystem='majority'

# Set random seeds
randomSeed = 42
seed_everything(randomSeed)

## LOAD THE DATA
## ==============================================================================================================
name = f'preprocessCombinations/hasBackground={hasBackground}-usesLargestBox={usesLargestBox}-segmentsMultiple={segmentsMultiple}-size=(119,119)'

def loadFromPickle(name):
    with open(f'{name}.pkl', 'rb') as fp:
        data = pkl.load(fp)
    return data

dataset = loadFromPickle(name)
# print('readingTemp Shape:', readingTemp.shape)    

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

consitencyCheck, instanceSize = checkShapesConsistent(dataset)
print('Sizes are all the same? ', consitencyCheck)
assert consitencyCheck
print(f'dataset shape:')
print(len(dataset), instanceSize)


# In[28]:


## SPLIT THE DATA
if sampleStrategy == 'underSampling':
    dataset = underSampleData(dataset)
    
# if sampleStrategy == 'overSampling':
#     trainData = overSampleData(trainFolders)



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

        ## Select and Train Model
        model, criterion, scheduler, optimizer = defineModel(learningRate=testInformation['learningRate'], weight_decay=testInformation['weight_decay'], 
                                                             model = testInformation['modelChosen'])
        model, criterion, device, history, endingEpoch = trainModel(model, criterion, scheduler, optimizer, trainData, valData, 
                                                                    patience=testInformation['patience'],numOfEpochs=testInformation['numOfEpochs'])

        saveResults('Tests/'+testInformation['testName']+f'/fold-{i+1}/', model, history, training_data_transforms, saveModel=False)

        #Evaluate perforamnce on test set
        confusionMatrixDisp, rocCurveDisplay, testingMetrics = evaluateModelOnTestSet('Tests/'+testInformation['testName']+f'/fold-{i+1}/', model, testData, criterion, device, 
                                                                                      testInformation['votingSystem'], testInformation['segmentsMultiple'], 
                                                                                      saveConfusionMatrix = False, showConfusionMatrix=False,
                                                                                      showROCCurve=False, saveROCCurve=False)
        
        plotTraining('Tests/'+testInformation['testName']+f'/fold-{i+1}/', '-', history, saveFigure=False, showResult=False)

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
    writeDictionaryToTxtFile('Tests/'+testInformation['testName']+'/kFoldsTestMetrics.txt',kFoldsTestMetrics, printLine=True)
    writeDictionaryToTxtFile('Tests/'+testInformation['testName']+'/testInformation.txt',testInformation, printLine=False)
    
    print('\n\n')
    # Plot training, confusion matrix, and roc curves for each fold as a single .png
    plotConfusionMatricies('Tests/'+testInformation['testName'], testInformation['testName'], confusion_matricies)
    plotROCCurves('Tests/'+testInformation['testName'], testInformation['testName'], rocCurves)
    plotTrainingPerformances('Tests/'+testInformation['testName'], testInformation['testName'], histories, saveFigure=True, showResult=True)

    appendMetricsToXLSX(testInformation['evalDetailLine'], testInformation['testName'], kFoldsTestMetrics, dataframe)

    #Make copies of the two scripts
    for filename in os.listdir():
        # Check if the file ends with .py
        if filename.endswith('.py'):
            # Copy the .py file
            shutil.copy(filename, 'Tests/'+testInformation['testName']+'/'+filename)


# In[ ]:


columns = ['TestName','RunData','PredictionSplits','Accuracy','F1Average','RecallAverage','PrecisionAverage','ROC-AUC','EndingEpoch','AccuracyData','F1Data','RecallData','PrecisionData','ROC-AUCData']
dataframePath='testResultsNew.xlsx'
sheetName = 'KFolds'
dataframe = pd.read_excel(dataframePath, sheetName,header=None, names=columns)
#addEvalDetailToModel(testInformation['evalDetailLine'],dataframe)
generateKFoldsValidation(testInformation, dataset, dataframe, k=5, trainingTransform=None) 
dataframe.to_excel(dataframePath, sheetName, index=False, header=False)


# In[12]:


# def generateKFoldsValidation(identifier,identifierValue, modelInformation, grouped2D, croppedSegmentsList, recistCriteria, cases, k=5,trainingTransform=None):

#     # Set the random seed for reproducibility
#     random.seed(0)
#     torch.manual_seed(0) 
    
#     #Keep history of values
#     confusion_matricies = []
#     histories = []
#     accuracies = []
#     f1s = []
#     recalls = []
#     predictionSplits = []
#     endingEpochs = []

#     print(recistCriteria)
#     ## =======================
#     ## For undersampling the 0 class
#     differenceIn0sTo1s = recistCriteria.count(0) - recistCriteria.count(1)
#     print('previous difference', differenceIn0sTo1s)
#     indiesToConsiderDropping = []
#     for i in range(len(recistCriteria)):
#         if recistCriteria[i] == 0:
#             indiesToConsiderDropping.append(i)
    
#     randomIndicies = random.sample(indiesToConsiderDropping, differenceIn0sTo1s)
#     croppedSegmentsList = np.delete(croppedSegmentsList,randomIndicies, axis=0)
#     recistCriteria = np.delete(recistCriteria,randomIndicies, axis=0).tolist()
#     cases = np.delete(cases,randomIndicies, axis=0).tolist()

#     differenceIn0sTo1s = recistCriteria.count(0) - recistCriteria.count(1)
#     print('New difference after undersampling', differenceIn0sTo1s)

#     ## =======================

    
#     if grouped2D: #if >100 then we are doing groupings of 2D images
#         stratifiedGroupFolds = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
#         stratifiedGroupFolds.get_n_splits(croppedSegmentsList, recistCriteria)
#         splits = enumerate(stratifiedGroupFolds.split(croppedSegmentsList, recistCriteria, cases))
        
#     else:
#         stratifiedFolds = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
#         stratifiedFolds.get_n_splits(croppedSegmentsList, recistCriteria)
#         splits = enumerate(stratifiedFolds.split(croppedSegmentsList, recistCriteria))
        
#     for i, (train_index, test_index) in splits:
        
#         #Set the name of the test
#         testName = f'{identifier}-{identifierValue}'
        
#         testPathName = 'Tests/'+testName+f'/foldn{i+1}'
#         print(f'{identifier}: foldn{i+1} RUN\n=========================================')
#         xTest, yTest = [croppedSegmentsList[i] for i in test_index], [recistCriteria[i] for i in test_index]
#         xTrain, yTrain = [croppedSegmentsList[i] for i in train_index], [recistCriteria[i] for i in train_index]
#         xVal, yVal = xTest, yTest # Set the validation set to the same as the testing set
#         #xVal, yVal = [croppedSegmentsList[i] for i in train_index[:len(xTest)]], [recistCriteria[i] for i in train_index[:len(yTest)]]
        

#         # ## Working with Numpy arrays
#         xTrain = np.array(xTrain) 
#         xTest = np.array(xTest)
#         xVal = np.array(xVal)
#         yTrain = np.array(yTrain)
#         yVal = np.array(yVal)
#         yTest = np.array(yTest)

#         ## ==============================================================
#         ## Using SMOTE
#         # smote = SMOTE(random_state=42)
#         # if len(xTrain.shape)==3:
#         #     oneDShape = xTrain[0].shape[0]*xTrain[0].shape[1]
            
#         # else:
#         #     print('xTrain shape',xTrain.shape)
#         #     oneDShape = xTrain[0].shape[0]*xTrain[0].shape[1]*xTrain[0].shape[2]

#         # singleShape = xTrain[0].shape

#         # print('xTrain reshape',xTrain.reshape(xTrain.shape[0],oneDShape).shape)
#         # xTrainSmote, yTrain = smote.fit_resample(xTrain.reshape(xTrain.shape[0],oneDShape), yTrain)
#         # if len(xTrain.shape)==3:
#         #     xTrain = xTrainSmote.reshape(xTrainSmote.shape[0], xTrain[0].shape[0],xTrain[0].shape[1])
#         # else:
#         #     xTrain = xTrainSmote.reshape(xTrainSmote.shape[0], xTrain[0].shape[0],xTrain[0].shape[1],xTrain[0].shape[2])

#         # print('xTrain after Smote', xTrain.shape)
#         # print('yTrain after Smote', yTrain.shape)
#         # from collections import Counter
#         # counter  = Counter(yTest)
#         # print('Splits for test Fold',sorted(counter.items()))

#         ## ==============================================================
        
#         # May or may not need this, def not needed for grouped2D=True
#         # xTrain = np.expand_dims(xTrain,axis=-1)
#         # xVal = np.expand_dims(xVal,axis=-1)
#         # xTest = np.expand_dims(xTest,axis=-1)

#         print('xTrain', xTrain.shape)
#         print('xVal', xVal.shape)
#         print('xTest', xTest.shape)

#         ## Get and save results for each fold        
#         confusionMatrix, history, accuracy, f1, recall, predictsTotal, endingEpoch = runModelFullStack(testPathName, testName, xTrain, yTrain, xVal, yVal, xTest, yTest, trainingTransform=trainingTransform, modelInformation=modelInformation) 

#         confusion_matricies.append(confusionMatrix)
#         histories.append(history)
#         accuracies.append(accuracy)
#         f1s.append(f1)
#         recalls.append(recall)
#         predictionSplits.append(predictsTotal)
#         endingEpochs.append(endingEpoch)
#         print('\n\n')

#     #assert False
#     # Calculate the average of the metrics for the kfolds of this transformation and save it
#     kFoldsTestMetrics = {'Prediction averages': averagePredictionTotals(predictionSplits), 'Accuracy':meanConfidenceInterval(accuracies), 'F1 Score':averageMultilabelMetricScores(f1s), 'Recall':averageMultilabelMetricScores(recalls)}
#     file = open('Tests/'+testName+'/kFoldsTestMetrics.txt','w')
#     for key, value in kFoldsTestMetrics.items():
#         file.write(f'{key}: {value}\n')
#         print(f'{key}: {value}')
#     file.close()

#     # Plot training and confusion matrix for each fold as a single .png
#     plotConfusionMatricies('Tests/'+testName, testName, confusion_matricies)
#     plotTrainingPerformances('Tests/'+testName, testName, histories, saveFigure=True, showResult=True)

#     # Append results to the xlsx file
#     appendMetricsToXLSX(testPathName, trainingTransform, meanConfidenceInterval(accuracies), averageMultilabelMetricScores(f1s), averageMultilabelMetricScores(recalls), averagePredictionTotals(predictionSplits), endingEpochs, modelInformation, dataframe)


# In[13]:


# #Make all transforms that I am going to test:
# transformsTested = {
#     #"0":None
#     #"20":generateTransform(RandomRotationValue=20, RandomElaticTransform=[20.0,2.0], brightnessConstant=20, contrastConstant=20, kernelSize=3, sigmaRange=(0.001,0.4)),
#     # "40":generateTransform(RandomRotationValue=40, RandomElaticTransform=[40.0,4.0], brightnessConstant=40, contrastConstant=40, kernelSize=3, sigmaRange=(0.001,0.8)),
#     # "60":generateTransform(RandomRotationValue=60, RandomElaticTransform=[60.0,6.0], brightnessConstant=60, contrastConstant=60, kernelSize=3, sigmaRange=(0.001,1.2)),
#     # "80":generateTransform(RandomRotationValue=80, RandomElaticTransform=[80.0,8.0], brightnessConstant=80, contrastConstant=80, kernelSize=3, sigmaRange=(0.001,1.6)),
#     #"100":generateTransform(RandomRotationValue=100, RandomElaticTransform=[100.0,10.0], brightnessConstant=100, contrastConstant=100, kernelSize=3, sigmaRange=(0.001,2.0)),
#     "defaults+50%":generateTransform(RandomRotationValue=50, RandomElaticTransform=[50.0,5.0], brightnessConstant=50.0, contrastConstant=50.0, kernelSize=3, sigmaRange=(0.1,2.0))    
# }

# ## Open the dataframe and add the evaluation details
# columns = ['name','numOfEpochs','batchSize','learningRate','dropoutRate','weight_decay', 'commandRan','RandomRotation','ElasticTransform','Brightness','Contrast','GaussianBlur','RandomHorizontalFlip','RandomVerticalFlip','PredictionAverage',
#             'AccuracyAverage','F1Average', 'RecallAverage','AccuracySTD','F1STD','RecallSTD','AccuracyData','F1Data','RecallData', 'EndingEpoch']


# # Generate the command ran for the test 
# commandRan = 'python'
# for details in sys.argv:
#     print('details',details)
#     stringArgs = ['evalDetailLine','modelChosen','votingSystem']
#     if details in stringArgs:
#         detailArray = details.split('=') 
#         details = f'{detailArray[0]}=\'{detailArray[1]}\''
#     commandRan += f' {details}'   
# print(commandRan)

# #Define patience
# patience = 10

# modelInformation = { 'learningRate': learningRate, 'dropoutRate': dropoutRate, 'batchSize': batchSize, 'numOfEpochs':numOfEpochs, 'weight_decay':weight_decay, 'commandRan': commandRan, 'model': modelChosen, 'patience':patience, 'votingSystem': votingSystem}
# # modelInformation = { 'learningRate': learningRate, 'dropoutRate': dropoutRate, 'batchSize': batchSize, 'numOfEpochs':1, 'weight_decay':weight_decay, 'commandRan': commandRan, 'model': modelChosen, 'votingSystem': votingSystem}

# # Run the tests
# for key, value in transformsTested.items():
#     generateKFoldsValidation(evalDetailLine,"-", croppedSegmentsList=croppedSegmentsList, recistCriteria=recistCriteria, cases=cases, k=5,trainingTransform=value, modelInformation = modelInformation, grouped2D=grouped2D)

# dataframe.to_excel(dataframePath, index=False, header=False)


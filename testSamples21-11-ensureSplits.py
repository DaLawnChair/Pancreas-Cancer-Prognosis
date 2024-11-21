#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # Convert to python script, remember to delete/comment the next line in the actual file
# ! jupyter nbconvert --to python twoClassClassification.ipynb --output testSamples21-11.py

# # Run the notebook in Simpson GPU server
# CUDA_VISIBLE_DEVICES=0 python testSamples2-8.py -batchSize=16 -epochs=100 -lr=0.001 -evalDetailLine="majourity voting on smote with 2 clases" -hasBackground=f -usesLargestBox=f -segmentsMultiple=12 -dropoutRate=0.2 -grouped2D=t -modelChosen='Small2D' -votingSystem='majority'
# CUDA_VISIBLE_DEVICES=1 python test14-8.py && CUDA_VISIBLE_DEVICES=1 python test14-8.py


# CUDA_VISIBLE_DEVICES=2 python testSamples11-11.py


# ### # Imports

# In[2]:


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


# In[3]:


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


# In[ ]:


def appendTestToDictionary(votingResults, testingMetrics,endingEpoch,history,confusionMatrixDisp,rocCurveDisplay, model):

    votingResults["accuracies"].append(testingMetrics['Accuracy'])
    votingResults["f1s"].append(testingMetrics['F1'])
    votingResults["recalls"].append(testingMetrics['Recall'])
    votingResults["predictionSplits"].append(testingMetrics['PredictionSplits'])
    votingResults["precisions"].append(testingMetrics['Precision'])
    votingResults["rocAucs"].append(testingMetrics['ROC-AUC'])
    votingResults["endingEpochs"].append(endingEpoch)
    
    votingResults["histories"].append(history)
    votingResults["confusion_matricies"].append(confusionMatrixDisp)
    votingResults["rocCurves"].append(rocCurveDisplay)

    votingResults["models"].append(model)
    return votingResults

def saveTestResults(votingResults, resultName, testInformation, dataframe, saveMaxAUCModel=True):
    kFoldsTestMetrics = {'Accuracy':meanConfidenceInterval(votingResults["accuracies"]), 
                        'F1':averageMultilabelMetricScores(votingResults["f1s"]), 
                        'Recall':averageMultilabelMetricScores(votingResults["recalls"]), 
                'PredictionSplits':averagePredictionTotals(votingResults["predictionSplits"]), 
                'Precision':averageMultilabelMetricScores(votingResults["precisions"]), 
                'ROC-AUC':meanConfidenceInterval(votingResults["rocAucs"]), 
                'endingEpochs':votingResults["endingEpochs"]}
    
    os.makedirs(resultName)
    # Write the test information and testvalues to files
    ##print(f"\n--------------------------------{testInformation['evalDetailLine']} -- AVERAGES --------------------------------")
    writeDictionaryToTxtFile(resultName+'/kFoldsTestMetrics.txt',kFoldsTestMetrics, printLine=True)
    writeDictionaryToTxtFile(resultName+'/testInformation.txt',testInformation, printLine=False)
    
    #print('\n\n')
    # Plot training, confusion matrix, and roc curves for each fold as a single .png
    plotConfusionMatricies(resultName, f"{testInformation['evalDetailLine']}", votingResults["confusion_matricies"])
    plotROCCurves(resultName, f"{testInformation['evalDetailLine']}", votingResults["rocCurves"])
    plotTrainingPerformances(resultName, f"{testInformation['evalDetailLine']}", votingResults["histories"], saveFigure=True, showResult=True)

    # save the model
    if saveMaxAUCModel:
        torch.save(votingResults["models"][np.argmax(votingResults['rocAucs'])].state_dict(), resultName+'/maxAUCModel.pt')

    appendMetricsToXLSX(testInformation['evalDetailLine'], testInformation['testName'], kFoldsTestMetrics, dataframe)


# In[ ]:


def generateKFoldsValidation(testInformation, dataset, dataframes, k=5):
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
#     'sampleStrategy': sampleStrategy,
#     'training_data_transforms': training_data_transforms
# }
    randomSeed = 42
    seed_everything(randomSeed)


    # Get the testing set
    patients = list(dataset.keys())
    labels = [dataset[patient]['label'] for patient in patients]
    
    # Split the dataset into k folds
    stratifiedFolds = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    stratifiedFolds.get_n_splits(patients, len(labels))
    splits = enumerate(stratifiedFolds.split(patients,labels))

    singleLargestVoting = {
        "accuracies": [],
        "f1s": [],
        "recalls": [],
        "predictionSplits": [],
        "precisions": [],
        "rocAucs": [],
        "endingEpochs": [],
        "histories": [],
        "confusion_matricies": [],
        "rocCurves": [],
        "models": []
    }
    
    averageVoting = {
        "accuracies": [],
        "f1s": [],
        "recalls": [],
        "predictionSplits": [],
        "precisions": [],
        "rocAucs": [],
        "endingEpochs": [],
        "histories": [],
        "confusion_matricies": [],
        "rocCurves": [],
        "models": []
    }
    majorityVoting = {
        "accuracies": [],
        "f1s": [],
        "recalls": [],
        "predictionSplits": [],
        "precisions": [],
        "rocAucs": [],
        "endingEpochs": [],
        "histories": [],
        "confusion_matricies": [],
        "rocCurves": [],
        "models": []
    }
    
    #print(f"\n\n====================Begin testing for {testInformation['evalDetailLine']}====================")

    originalDataset = copy.deepcopy(dataset)

    for i, (trainIndicies, valIndicies) in splits:
        seed_everything(randomSeed)
        dataset = copy.deepcopy(originalDataset)

        patients = list(dataset.keys())
        trainFolders = [patients[i] for i in trainIndicies]
        valFolders = [patients[i] for i in valIndicies]
        testFolders = [patients[i] for i in valIndicies]

        if testInformation['sampleStrategy'] == 'overSampling':
            dataset, trainFolders = oversampleData(dataset, trainFolders)
        elif testInformation['sampleStrategy'] == 'underSampling':
            dataset, trainFolders = underSampleData(dataset, trainFolders)
                
        trainData, valData, testData, training_data_transforms = convertDataToLoaders(trainFolders, valFolders, testFolders, dataset, 
                                                                                      testInformation['modelChosen'], testInformation['grouped2D'], 
                                                                                      testInformation['segmentsMultiple'], 
                                                                                      training_data_transforms = testInformation['training_data_transforms'], 
                                                                                      batchSize=testInformation['batchSize'])
        

        #print(f"\n--------------------------------{testInformation['evalDetailLine']} -- Fold #{i+1}--------------------------------")

        # f = open(f"Testcheck{i}-{testInformation['modelChosen']}.txt","r")

        # trainDataPrint = f"Train patients ({len(trainFolders)}): {trainFolders}"
        # valDataPrint = f"validation patients ({len(testFolders)}): {valFolders}"
        # for line in f.readline():
        #     if line.strip() != trainDataPrint or line.strip() != valDataPrint:
        #         print("difference in strings")
        #         assert False
        print(f"Train patients ({len(trainFolders)}):", trainFolders)
        print(f"validation patients ({len(testFolders)}):", valFolders)
        print(f"test patients ({len(testFolders)}):", "same as valFolders" if testFolders==valFolders else testFolders)
        
        with open(f"Testcheck{i}-{testInformation['modelChosen']}.txt", 'w') as f:
            print(f"Train patients ({len(trainFolders)}):", trainFolders, file=f)
            print(f"validation patients ({len(testFolders)}):", valFolders,file=f)     
            print(f"test patients ({len(testFolders)}):", "same as valFolders" if testFolders==valFolders else testFolders, file=f)
            
        #f.write(f"Train patients ({len(trainFolders)}): {trainFolders}".encode('utf-8', 'ignore'))
        #f.write(f"validation patients ({len(testFolders)}): {valFolders}".encode('utf-8', 'ignore'))
        
        f.close()
        continue

        resultName = 'Tests/'+testInformation['testName']+'/'+testInformation['evalDetailLine']
        resultNameWithFold = resultName+f'/fold-{i+1}/'

        saveResults(resultNameWithFold, model, history, training_data_transforms, saveTrainTransforms=False, saveModel=False)

        ## Select and Train Model
        model, criterion, scheduler, optimizer = defineModel(learningRate=testInformation['learningRate'], weight_decay=testInformation['weight_decay'], 
                                                             model = testInformation['modelChosen'])
        model, criterion, device, history, endingEpoch = trainModel(model, criterion, scheduler, optimizer, trainData, valData, 
                                                                    patience=testInformation['patience'],numOfEpochs=testInformation['numOfEpochs'])

        ##Evaluate perforamnce on test set
        # =============================================================================

        if testInformation['votingSystem']=='singleLargest':
            confusionMatrixDisp, rocCurveDisplay, testingMetrics = evaluateModelOnTestSet(resultNameWithFold, model, testData, criterion, device, 
                                                                                'singleLargest', 1, 
                                                                                saveConfusionMatrix = False, showConfusionMatrix=False,
                                                                                showROCCurve=False, saveROCCurve=False)

            plotTraining(resultNameWithFold, '-', history, saveFigure=False, showResult=False)
            singleLargestVoting = appendTestToDictionary(singleLargestVoting, testingMetrics, endingEpoch,history,confusionMatrixDisp,rocCurveDisplay, model)

        else:
            confusionMatrixDisp, rocCurveDisplay, testingMetrics = evaluateModelOnTestSet(resultNameWithFold, model, testData, criterion, device, 
                                                                                        'average', testInformation['segmentsMultiple'], 
                                                                                        saveConfusionMatrix = False, showConfusionMatrix=False,
                                                                                        showROCCurve=False, saveROCCurve=False)
            
            plotTraining(resultNameWithFold, '-', history, saveFigure=False, showResult=False)
            averageVoting = appendTestToDictionary(averageVoting, testingMetrics, endingEpoch,history,confusionMatrixDisp,rocCurveDisplay, model)


            confusionMatrixDisp, rocCurveDisplay, testingMetrics = evaluateModelOnTestSet(resultNameWithFold, model, testData, criterion, device, 
                                                                                        'majority', testInformation['segmentsMultiple'], 
                                                                                        saveConfusionMatrix = False, showConfusionMatrix=False,
                                                                                        showROCCurve=False, saveROCCurve=False)
            plotTraining(resultNameWithFold, '-', history, saveFigure=False, showResult=False)
            majorityVoting = appendTestToDictionary(majorityVoting, testingMetrics, endingEpoch,history,confusionMatrixDisp,rocCurveDisplay, model)





    # if testInformation['votingSystem']== 'singleLargest':
    #     saveTestResults(singleLargestVoting, resultName+'/singleLargestVoting', testInformation, dataframes['singleLargest.xlsx'])
    # else:
    #     saveTestResults(averageVoting, resultName+'/averageVoting', testInformation, dataframes['average.xlsx'])
    #     saveTestResults(majorityVoting, resultName+'/majorityVoting', testInformation, dataframes['majority.xlsx'])

    # #Make copies of the two scripts
    # for filename in os.listdir():
    #     # Check if the file ends with .py
    #     if filename.endswith('.py'):
    #         # Copy the .py file
    #         shutil.copy(filename, resultName+'/'+filename)


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
            #print(f"Error in shape at index {i} with shape {data[keys[i]]['images'].shape}")
            return False , size
    return True , size
    


def getTestingSet(dataset):
    
    seed_everything(42)
    # choose 14 samples of the 89 samples, for about 85% split between train and validation and 15% for test
    # testSamples = random.sample(list(dataset.keys()), 14) 
    #print('Testing set samples (key, classification):')
    
    # Have defined our test set from the above, hardcoding for validity
    testSet = {"CASE616": torch.tensor(1, dtype=torch.float32),
                "CASE472": torch.tensor(1, dtype=torch.float32),
                "CASE251": torch.tensor(1, dtype=torch.float32),
                "CASE539": torch.tensor(1, dtype=torch.float32),
                "CASE531": torch.tensor(1, dtype=torch.float32),
                "CASE520": torch.tensor(0, dtype=torch.float32),
                "CASE480": torch.tensor(0, dtype=torch.float32),
                "CASE471": torch.tensor(1, dtype=torch.float32),
                "CASE630": torch.tensor(0, dtype=torch.float32),
                "CASE596": torch.tensor(1, dtype=torch.float32),
                "CASE467": torch.tensor(1, dtype=torch.float32),
                "CASE604": torch.tensor(1, dtype=torch.float32),
                "CASE569": torch.tensor(1, dtype=torch.float32),
                "CASE254": torch.tensor(1, dtype=torch.float32)}
    
    for key in testSet:
        pass
        #print(key,dataset[key]['label'])
    
    return testSet

def getDataset(testInformation):
    # Set random seeds
    randomSeed = 42
    seed_everything(randomSeed)

    ## LOAD THE DATA
    ## ==============================================================================================================
    name = f"preprocessCombinations/hasBackground={testInformation['hasBackground']}-usesLargestBox={testInformation['usesLargestBox']}-segmentsMultiple={testInformation['segmentsMultiple']}-size=(119,119)"
    
    dataset = loadFromPickle(name)
    consitencyCheck, instanceSize = checkShapesConsistent(dataset)
    #print('Sizes are all the same? ', consitencyCheck)
    assert consitencyCheck
    #print(f'dataset shape:')
    #print(len(dataset), instanceSize)
    
    return dataset


# In[ ]:


columns = ['TestName','RunData','PredictionSplits','Accuracy','F1','Recall','Precision','ROC-AUC','EndingEpoch','AccuracyData','F1Data','RecallData','PrecisionData','ROC-AUCData']
#sheetName = 'KFolds'

def getDataframes():
    for sheetName in ['singleLargest.xlsx', 'average.xlsx', 'majority.xlsx']:
        if not os.path.exists(sheetName): # Creates datasheet if it doesn't exist
            dataframe = pd.DataFrame(columns=columns)
            dataframe.to_excel(sheetName, index=False, engine='openpyxl')

    dataframes = {
        "singleLargest.xlsx": pd.read_excel('singleLargest.xlsx',header=None, names=columns, engine='openpyxl'),
        "average.xlsx": pd.read_excel('average.xlsx',header=None, names=columns, engine='openpyxl'),
        "majority.xlsx": pd.read_excel('majority.xlsx',header=None, names=columns, engine='openpyxl')
    }
    return dataframes 


def gridSearch(testInformation):
    for key, value in testInformation.items():
        #print(f'{key}: {value}')
        pass

    dataframes = getDataframes()
    #addEvalDetailToModel(testInformation['evalDetailLine'],dataframe)

    dataset = getDataset(testInformation)
    generateKFoldsValidation(testInformation, dataset, dataframes, k=5) 

    for sheetName, dataframe in dataframes.items():
        dataframe.to_excel(sheetName, index=False, header=False)

def setGridSearchParams():
    # Unchanging Data parameters
    numOfEpochs = 50
    hasBackground = True
    usesLargestBox = True
    # votingSystem = 'multiVoting' #average, singleLargest, majority, multiVoting
    sampleStrategy = 'normal' # 'underSampling', 'overSampling', 'normal' 

    experimentName = f'testSamples21-11--normal'
    training_data_transforms = None
    # training_data_transforms = transforms.Compose([
    #     transforms.RandomRotation(degrees=0.85),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomVerticalFlip(p=0.5)
    # ]) 

    for segmentsMultiple in [3]:

        if segmentsMultiple==1:
            votingSystem = 'singleLargest'
            grouped2D = False 
        else:
            votingSystem = 'multiVoting'
            grouped2D = True
        for modelChosen in ['InceptionV3Small2D', 'ResNet50Small2D','VGG16Small2D','XceptionSmall2D']: #, 'ResNet50Small2D','InceptionV3Small2D','VGG16Small2D', 'XceptionSmall2D']:

            ## Differentiate the start of the test with a line            
            dataframes = getDataframes()
            if votingSystem=='singleLargest':
                addEvalDetailToModel(f"{modelChosen}, segments={segmentsMultiple}, voting=singleLargest", dataframes['singleLargest.xlsx'])
            if votingSystem=='average' or votingSystem=='multiVoting':
                addEvalDetailToModel(f"{modelChosen}, segments={segmentsMultiple}, voting=average", dataframes['average.xlsx'])
            if votingSystem=='majority' or votingSystem=='multiVoting':
                addEvalDetailToModel(f"{modelChosen}, segments={segmentsMultiple}, voting=majority", dataframes['majority.xlsx'])

            for sheetName, dataframe in dataframes.items():
                dataframe.to_excel(sheetName, index=False, header=False)
            
            # Beging hyperparemeterizing
            for learningRate in [0.001]:
                for weight_decay in [0.01]:
                    for batchSize in [8]:
                        for patience in [5]:
                            testName = f'{experimentName}/{modelChosen}-segmentsMultiple={segmentsMultiple}'
                            evalDetailLine = f"-modelChosen={modelChosen}-lr={learningRate}-weight_decay={weight_decay}-batchSize={batchSize}-patience={patience}"
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
                                'sampleStrategy': sampleStrategy,
                                'training_data_transforms': training_data_transforms
                            }
                            gridSearch(testInformation)

setGridSearchParams()


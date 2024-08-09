#!/usr/bin/env python
# coding: utf-8

# In[55]:


# # Convert to python script, remember to delete/comment the next line in the actual file
# ! jupyter nbconvert --to python imageSegmentation2Classes.ipynb --output testSamples4-8-undersampled.py

# # Run the notebook in Simpson GPU server
# CUDA_VISIBLE_DEVICES=0 python testSamples2-8.py -batchSize=16 -epochs=100 -lr=0.001 -evalDetailLine="majourity voting on smote with 2 clases" -hasBackground=f -usesLargestBox=f -segmentsMultiple=12 -dropoutRate=0.2 -grouped2D=t -modelChosen='Small2D' -votingSystem='majority'


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
import pickle

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
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Oversampling
from imblearn.over_sampling import SMOTE


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


# ## Get the data from the .xsxl file

# In[41]:


columns = ['TAPS_CaseIDs_PreNAT','RECIST_PostNAT', 'Slice_Thickness']
data = pd.read_excel('PDAC-Response_working.xlsx', header=None,names = columns)
data.drop(0, inplace=True) # Remove the header row
data=data.sort_values(by=['TAPS_CaseIDs_PreNAT'])
data.drop(data[data['Slice_Thickness'] < 6].index, inplace = True)


# # Get the entire datasheet
cases = list(data['TAPS_CaseIDs_PreNAT'])
recistCriteria = list(data['RECIST_PostNAT'])

sliceThickness = list(data['Slice_Thickness'])
sliceThickness.sort()
print(f'Number of patients: {len(sliceThickness)}\nSlice range: {min(sliceThickness)}-{max(sliceThickness)}')


print(f'Distribution of classes previously:')
counter = Counter(recistCriteria)
print(sorted(counter.items()))

# recistNames = [ 'Progressive \nDisease','Stable \nDisease','Partial \nResponse'] 
# names = ['Not evaluated/\nNot recorded', 'Progressive \nDisease','Stable \nDisease','Partial \nResponse','Complete \nResponse'] 

## Combine the progressive disease and stable disease
# Dataset does not have 0 or else cases
for i in range(len(recistCriteria)):
    if recistCriteria[i] == 0: # Not evaluated
        recistCriteria[i] = 0
    elif recistCriteria[i] == 1: # Progressive disease
        recistCriteria[i] = 1
    elif recistCriteria[i] == 2: # Stable disease
        recistCriteria[i] = 1
    elif recistCriteria[i] == 3: # Partial Response
        recistCriteria[i] = 2
    elif recistCriteria[i] == 4: # Complete Response
        recistCriteria[i] = 2 
    else: # Other
        recistCriteria[i] = 0    

print(f'Distribution of classes after combining progressive disease with stable disease:')
counter = Counter(recistCriteria)
print(sorted(counter.items()))

recistCriteria = [i-1 for i in recistCriteria] # Make the classes 0, 1
print('Make recistCriteria 0 and 1:',recistCriteria)


# In[5]:


# Displaying segments
#==========================================================================================

def displayCroppedSegmentations(croppedSegment):
    print(f'CroppedSegment shape: {croppedSegment.shape}')
    # Display the segmented image slices 

    columnLen = 10
    rowLen = max(2,croppedSegment.shape[0] // columnLen + 1) 
    figure,axis = plt.subplots( rowLen, columnLen, figsize=(10, 10))
    
    rowIdx = 0
    for idx in range(croppedSegment.shape[0]):        
        if idx%columnLen == 0 and idx>0:
            rowIdx += 1        
        # axis[rowIdx][idx%columnLen].imshow(croppedSegment[idx,:,:] , cmap="gray", vmin = 40-(350)/2, vmax=40+(350)/2)
        axis[rowIdx][idx%columnLen].imshow(croppedSegment[idx,:,:] , cmap="gray")

        axis[rowIdx][idx%columnLen].axis('off')

    # Turn off the axis of the rest of the subplots
    for i in range(idx+1, rowLen*columnLen):
        if i%columnLen == 0:
            rowIdx += 1
        axis[rowIdx][i%columnLen].axis('off')
    
    plt.show()


def displayOverlayedSegmentations(segmentedSlices, augmented_whole, augmented_segment):
    # Display the segmented image slices 
    columnLen = 10
    rowLen = max(2,len(segmentedSlices) // columnLen + 1) 
    figure,axis = plt.subplots( rowLen, columnLen, figsize=(10, 10))
    rowIdx = 0
    for idx in range(len(segmentedSlices)):        
        if idx%columnLen == 0 and idx>0:
            rowIdx += 1
        axis[rowIdx][idx%columnLen].imshow(augmented_whole[segmentedSlices[idx],:,:], cmap="gray")
        axis[rowIdx][idx%columnLen].imshow(augmented_segment[segmentedSlices[idx],:,:], cmap="Blues", alpha=0.75)
        axis[rowIdx][idx%columnLen].axis('off')

    # Turn off the axis of the rest of the subplots
    for i in range(idx+1, rowLen*columnLen):
        if i%columnLen == 0:
            rowIdx += 1
        axis[rowIdx][i%columnLen].axis('off')
    plt.show()


# ## Perform preprocessing on multiple images
# 

# In[9]:


#ADD argparser
import argparse
import sys
print('Current System:',sys.argv[0])


# python testSamples2-8.py -batchSize=8 -epochs=100 -lr=0.001 -evalDetailLine="majourity voting on new data" -hasBackground=f -usesLargestBox=f -segmentsMultiple=12 -dropoutRate=0.2 -grouped2D=t -modelChosen='Small2D' -votingSystem='majority'


#Check if we are using a notebook or not
if 'ipykernel_launcher' in sys.argv[0]:
    batchSize = 8
    numOfEpochs = 100
    evalDetailLine = ""
    learningRate = 0.001
    hasBackground = False
    usesLargestBox = True
    segmentsMultiple = 6
    dropoutRate = 0.2
    grouped2D = True
    weight_decay = 0.01
    modelChosen = 'Small2D' #Large2D, Small2D
    votingSystem = 'majourity'

else:
    parser = argparse.ArgumentParser(description="Model information")
    parser.add_argument('-batchSize', type=int, help='batch size', default=8)
    parser.add_argument('-epochs', type=int, help='Number of Epochs', default=100)
    parser.add_argument('-lr', type=float, help='Learning Rate', default=0.001)
    parser.add_argument('-evalDetailLine', type=str, help='Details of the evaluation', default='')
    parser.add_argument('-hasBackground', type=str, help='Whether to use the background (t to, f to not)', default='f')
    parser.add_argument('-usesLargestBox', type=str, help='Where to use the size of the largest box (t) or independent tumor boxes (f)', default='t')
    parser.add_argument('-segmentsMultiple', type=int, help='Segments a # of slices, 1 by default', default=1)
    parser.add_argument('-dropoutRate', type=float, help='Dropout rate for the model', default=0.2)
    parser.add_argument('-grouped2D', type=str, help='Grouping the 3D scans as individual 2D images', default='f')
    parser.add_argument('-weightDecay', type=float, help='Weight Decay for the model', default=0.01)
    parser.add_argument('-modelChosen', type=str, help='Selected Model', default='Large2D')
    parser.add_argument('-votingSystem', type=str, help='Either \'majority\' voting, \'average\' voting, or \'singleLargest\' ', default='majority')
    
    args = parser.parse_args()
    
    batchSize = args.batchSize
    numOfEpochs = args.epochs
    evalDetailLine = args.evalDetailLine
    learningRate = args.lr
    hasBackground = True if args.hasBackground=='t' else False
    usesLargestBox = True if args.usesLargestBox=='t' else False
    segmentsMultiple = args.segmentsMultiple
    dropoutRate = args.dropoutRate
    grouped2D = True if args.grouped2D=='t' else False
    weight_decay = args.weightDecay
    modelChosen = args.modelChosen
    votingSystem = args.votingSystem


print(f'BatchSize: {batchSize}, Epochs: {numOfEpochs}, Learning Rate: {learningRate}, Eval Detail Line: {evalDetailLine}, Has Background: {hasBackground}, Uses Largest Box: {usesLargestBox}, Segments Multiple: {segmentsMultiple}, \
      Dropout Rate: {dropoutRate}, Grouped2D: {grouped2D}, weightDecay: {weight_decay}, modelChosen: {modelChosen}, votignSystem: {votingSystem}')  


# In[11]:


#Get a saved copy of the dataset
name = f'hasBackground={hasBackground}-usesLargestBox={usesLargestBox}-segmentsMultiple={segmentsMultiple}-size=(119,119)'        
# name = f'hasBackground=True-usesLargestBox=False-segmentsMultiple=1'
croppedSegmentsList = np.load(f'preprocessCombinations/{name}.npy')
    
print('croppedSegmentsList Shape:', croppedSegmentsList.shape)   
print('Single item Shape:', croppedSegmentsList[0].shape)   

        


# In[12]:


# figure,axis = plt.subplots(1,len(croppedSegmentsList),figsize=(200,100))
# for idx in range(len(croppedSegmentsList)):        
#     axis[idx].imshow(croppedSegmentsList[idx], cmap="gray")
#     axis[idx].axis('off')
# plt.savefig('padding10.png')
# # plt.show()


# # Data augmentation

# In[13]:


# Getting information about the transformations
def generateTransform(RandomHorizontalFlipValue=0.5,RandomVerticalFlipValue=0.5, RandomRotationValue=50, RandomElaticTransform=[0,0], brightnessConstant=0, contrastConstant=0, kernelSize=3, sigmaRange=(0.1,1.0)):
    training_data_transforms = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=RandomHorizontalFlipValue),
        transforms.RandomVerticalFlip(p=RandomVerticalFlipValue),
        #transforms.RandomRotation(degrees=RandomRotationValue),
        #transforms.ElasticTransform(alpha=RandomElaticTransform[0], sigma=RandomElaticTransform[1]),
        #transforms.ColorJitter(brightnessConstant, contrastConstant),
        #transforms.GaussianBlur(kernel_size = kernelSize, sigma=sigmaRange),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 
    return training_data_transforms


def getTransformValue(transform, desiredTranform, desiredTranformValue):
    if transform==None or desiredTranform==None or desiredTranformValue==None:
      return None
    for t in transform.transforms:
        if isinstance(t, desiredTranform):
            return t.__getattribute__(desiredTranformValue)
    return None



# In[14]:


# For 2D images:
# ## Working with pytorch tensors
class PatientData(Dataset):
    def __init__(self, image, classifications, transform=None):
        self.data = image
        # Convert classification to torch tensor
        temp = []
        for classification in classifications:
            convert = torch.tensor(classification, dtype=torch.int64) # casting to long
            convert = convert.type(torch.float32)
            temp.append(convert)
            
        self.classification = temp
        self.transform = transform

    def __len__(self):
        return len(self.data)* segmentsMultiple

    def __getitem__(self, idx):
        if grouped2D==False:
            image = self.data[idx]
            label = self.classification[idx]
        else:
            patient_idx = idx // segmentsMultiple
            slice_idx = idx % segmentsMultiple
            image = self.data[patient_idx][slice_idx]
            label = self.classification[patient_idx]

        # Convert to RGB
        image = Image.fromarray((image * 255).astype(np.uint16))
        image = image.convert("RGB")

        # Apply augmentations if there are any
        if self.transform:
            image = self.transform(image)#.type(torch.float)

        return image, label


def convertDataToLoaders(xTrain, yTrain, xVal, yVal, xTest, yTest, training_data_transforms = None, batchSize=8):
    ## Sample the data with 75% of the training set 
    # TrainBalancedSampler = WeightedRandomSampler(weightsForClasses, len(yTrain)//2+len(yTest)//4)
    
    # Testing data tranfrom, should be just the plain images
    testing_data_transforms = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize(224),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 

    # Use the same default training transform as the testing transform if not specified
    if training_data_transforms == None:
        print('Training transform is None, using default transforms')
        training_data_transforms = testing_data_transforms
        
    # Convert the testing sets to data loaders
    trainingData = PatientData(xTrain, yTrain, transform=training_data_transforms)
    trainingData = DataLoader(trainingData, batch_size=batchSize, shuffle=True)#, sampler= TrainBalancedSampler)

    validationData = PatientData(xVal, yVal, transform=testing_data_transforms)
    validationData = DataLoader(validationData, batch_size=batchSize, shuffle=False)

    testingData = PatientData(xTest, yTest, transform=testing_data_transforms)
    testingData = DataLoader(testingData, batch_size=batchSize, shuffle=False)

    return trainingData, validationData, testingData, training_data_transforms


# In[15]:


# # For 3D images:

# import torchio

# # ## Working with pytorch tensors
# class TorchDataset(Dataset):
#     def __init__(self, images, classifications, transform=None):
#         self.data = images
#         temp = []
#         for classification in classifications:
#             convert = torch.tensor(classification, dtype=torch.int64) # casting to long
#             convert = convert.type(torch.float32)
#             temp.append(convert)
            
#         self.classification = temp
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         images = self.data[idx]
#         # Normalize the images
#         images = (images - np.min(images)) / (np.max(images) - np.min(images))


#         images = torch.from_numpy(images)
#         images = torch.FloatTensor(images)

#         # images = torch.from_numpy(images)
#         print(type(images))
        

#         label = self.classification[idx]
#         # Convert sample to a tensor
#         # sample = torch.tensor(sample, dtype=torch.float32).permute(2, 0, 1)  # Change shape to (C, H, W)

#         if self.transform:
#             images = self.transform(images)
        
#         return images, label
    

# # Set the random seed for reproducibility
# random.seed(0)
# torch.manual_seed(0) 

# ## Sample the data with 75% of the training set 
# # TrainBalancedSampler = WeightedRandomSampler(weightsForClasses, len(yTrain)//2+len(yTest)//4)

# # Define data augmentation transforms
# training_data_transforms = torchio.Compose([
#     torchio.RandomFlip(axes=('Left','Right'), flip_probability=RandomVerticalFlipProbablility),
#     torchio.RandomFlip(axes=('Anterior','Posterior'), flip_probability=RandomHorizontalFlipProbablility),
#     torchio.RandomNoise(std=(0, 0.1)),
#     torchio.RandomBlur(std=(0, 1))
# ]) 
# # testing_data_transforms = transforms.Compose([
# #     transforms.ToPILImage(),
# #     transforms.ToTensor()
# # ]) 


# # Convert the testing sets to data loaders
# trainingData = TorchDataset(xTrain, yTrain, transform=training_data_transforms)

# trainingData = DataLoader(trainingData, batch_size=batchSize, shuffle=False)#, sampler= TrainBalancedSampler)

# testingData = TorchDataset(xTest, yTest, transform=None)# testing_data_transforms)
# testingData = DataLoader(testingData, batch_size=batchSize, shuffle=False)


# ### Define Model and training

# In[16]:


class Small2D(torch.nn.Module):
    def __init__(self, dropoutRate=0.2):
        super(Small2D, self).__init__()

        #Resnet50 as first layer
        self.model = models.resnet50(weights='IMAGENET1K_V2')
        
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x 
    
def defineModel(dropoutRate=0.2,learningRate=0.001, weight_decay=0.01, model = 'Small2D'):
    if model == 'Small2D':
        model = Small2D(dropoutRate)
    elif model == 'Large2D':
        model = Large2D(dropoutRate)
    elif model == 'foundationSimple':
        model = SmallFoundation(dropoutRate)
    
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learningRate, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    return model, criterion, scheduler, optimizer


# In[17]:

def train(model, loader, criterion, scheduler, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device).view(-1, 1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predicted = outputs > 0.5
        total += labels.size(0)
        running_loss += loss.item() * inputs.size(0)
        correct += torch.sum(predicted == labels.data).item()
        
    scheduler.step()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


class EarlyStopping:
    def __init__(self, patience=5, minDelta=0):
        self.patience = patience
        self.minDelta = minDelta
        self.counter = 0
        self.minValLoss = float('inf')
        
    def earlyStoppingCheck(self, currValLoss):
        if np.isnan(currValLoss):
            return True
        if currValLoss < self.minValLoss:
            self.minValLoss = currValLoss
            self.counter = 0
        elif currValLoss >= self.minValLoss + self.minDelta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



def evaluateGroupVoting(model, loader, criterion, device):
    model.eval()
    predictions = []
    probabilities = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1, 1)

            outputs = model(inputs)

            predicted = outputs > 0.5

            probabilities.append(outputs.cpu().numpy())
            predictions.append(predicted.cpu().numpy())
    
    # Ignores grouped voting
    if votingSystem=='singleLargest' or segmentsMultiple==1:
        return predictions

    ## Grouping predictions by patient
    ## ==============================================================
    all_probs = np.concatenate(probabilities, axis=0)

    # Grouping predictions by patient
    patient_probs = [] # The confidence of the model
    patient_labels = [] # the label given by the model. >=0.5 is 1, <0.5 is 0
     
    ## Get the classification based from the patient based on ...
    for i in range(0, len(all_probs), segmentsMultiple):
        # Get probabilties and labels for the patient
        patient_prob = all_probs[i:i + segmentsMultiple]
        patient_pred_labels = [pred>=0.5 for pred in patient_prob]

        ## Average all confidences to get the highest probability label, used as a tie breaker
        patient_prob = np.mean(patient_prob) 
        prob_label_max = patient_prob >= 0.5 
        
        ## MAJOURITY VOTING
        ##==============================================================
        # Get counts for the labels 
        if votingSystem=='majority':
            uniqueLabels, label_counts = np.unique(patient_pred_labels, return_counts=True)
            
            majorityVote = uniqueLabels[np.argmax(label_counts)]
            patient_labels.append(np.array([majorityVote]))
        ##==============================================================
        ## AVERAGE
        ##==============================================================
        elif votingSystem=='average':
            patient_probs.append(np.array(patient_prob))
            patient_labels.append(np.array([prob_label_max]))
        ##==============================================================

    return patient_labels


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1, 1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            predicted = outputs > 0.5
            total += labels.size(0)
            running_loss += loss.item() * inputs.size(0)
            correct += torch.sum(predicted == labels.data).item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, predictions


# In[19]:


def trainModel(model, criterion, scheduler, optimizer, trainingData, validationData, numOfEpochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using this device:', device)
    #Send the model to the same device that the tensors are on
    model.to(device)

    
    earlyStopping = EarlyStopping(patience=5, minDelta=0)
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    for epoch in range(numOfEpochs):
        #Train model
        curTrainLoss, curTrainAcc = train(model, trainingData, criterion, scheduler, optimizer, device)
        print(f"Epoch {epoch+1}/{numOfEpochs}")
        print(f"Train Loss: {curTrainLoss:.4f}, Train Acc: {curTrainAcc:.4f}")
        #Evaluate on validation set
        curValLoss, curValAcc, _ = evaluate(model, validationData, criterion, device)    
        print(f"Val Loss: {curValLoss:.4f}, Val Acc: {curValAcc:.4f}")

        #Append metrics to lists
        train_loss.append(curTrainLoss)
        train_acc.append(curTrainAcc)
        val_loss.append(curValLoss)
        val_acc.append(curValAcc)

        #Check for early stopping conditions
        if earlyStopping.earlyStoppingCheck(curValLoss):
            print(f'Early stopping - Val loss has not decreased in {earlyStopping.patience} epochs. Terminating training at epoch {epoch+1}.')
            break

    history = {'train_loss':train_loss, 'train_acc':train_acc, 'val_loss':val_loss, 'val_acc':val_acc}
    return model, criterion, device, history, epoch


# In[20]:


# def read_history_from_pickle(testPathName):
#     with open(testPathName+'/history.pkl', 'rb') as fp:
#         history = pickle.load(fp)
#     return history

# #Read history
# history = read_history_from_pickle(testPathName)

# # Load and evalaute the model
# modelWeightPath = testPathName+'/model.pt'
# model = ResNet50ClassificaitonModel()
# model.load_state_dict(torch.load(modelWeightPath))

#Send the model to the device used
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using this device:', device)
# #Send the model to the same device that the tensors are on
# model.to(device)



# ### Save results
# 

# In[21]:


# Save the contents of this test

def saveResults(testPathName, model, history, training_data_transforms, saveModel=True):
    os.makedirs(testPathName, exist_ok=True)

    #Save history as pickle
    with open(testPathName+'/history.pkl', 'wb') as fp:
        pickle.dump(history, fp)

    # Save weigths of model
    if saveModel:
        torch.save(model.state_dict(), testPathName+'/model.pt')

    # Save transformations for easy access
    f = open(testPathName + '/training_data_transforms.txt', 'w')
            
    for line in training_data_transforms.__str__():
        f.write(line)
    f.close()



# ### Evaluate performance on the testing set

# In[22]:


def convertTensorToPredictions(tensorList, dictionary):
    """Converts a list of tensors/ndarrays into a single list and then feeds those values into a given dictionary."""
    values=[]
    if grouped2D==True:
        for batch in tensorList:
            batchValues = batch.tolist()        
            values+= batchValues
        for value in values:
            dictionary[value]+=1
    else:
        values=[]#tensorList
        for batch in tensorList:
            batchValues = batch.tolist()      
            values.extend(batchValues)
        for i in range(len(values)):
            values[i] = values[i][0]
        print('tensorlist', tensorList)
        print('values', values)
        
        print(values)
        for value in values:
            dictionary[value]+=1
    return values, dictionary

# def convertTensorToPredictions(tensorList, dictionary):
#     """Converts a list of tensors/ndarrays into a single list and then feeds those values into a given dictionary."""
#     values=[]
#     for batch in tensorList:
#         batchValues = batch.tolist()      
#         values.extend(batchValues)
#     for i in range(len(values)):
#         values[i] = values[i][0]
#     print('tensorlist', tensorList)
#     print('values', values)
    
#     print(values)
#     for value in values:
#         dictionary[value]+=1
#     return values, dictionary

def evaluateModelOnTestSet(testPathName, model, testingData, criterion, device, saveConfusionMatrix = True, showConfusionMatrix=True):
    predictions = evaluateGroupVoting(model, testingData, criterion, device)

    predicts, predictsTotal = convertTensorToPredictions(predictions, {0:0,1:0})

    # Get the correct answers
    originalLabelsTemp = []
    ans = []
    
    for _, labels in testingData:
      originalLabelsTemp.extend(labels.tolist())
    
    for i in range(len(originalLabelsTemp)):  
        if i%segmentsMultiple==0:
            ans.append(originalLabelsTemp[i] >=0.5)
        i+=1
 
    ansTotal = dict(zip([0,1],[ans.count(False),ans.count(True)]))

    print('ans length',len(ans))

    # Test metrics
    print('---------------------------------------\nTesting Metrics')
    
    accuracy = accuracy_score(ans, predicts)
    f1 = list(f1_score(ans, predicts, average=None))  # Use 'weighted' for multiclass classification
    recall = list(recall_score(ans, predicts, average=None))  # Use 'weighted' for multiclass classification

    testingMetrics = {'Predictions split': predictsTotal, 'Answers split': ansTotal, 'Predictions': predicts, 'Answers    ': ans,  'Accuracy':accuracy, 'F1 Score':f1, 'Recall':recall}

    file = open(testPathName+'/testingMetrics.txt','w')
    for key, value in testingMetrics.items():
        file.write(f'{key}: {value}\n')
    file.close()

    for key, value in testingMetrics.items():
        print(f'{key}: {value}')


    print('---------------------------------------\nConfusion Matrix:')
    # Confusion Matrix
    result = confusion_matrix(ans,predicts,normalize='pred')
    disp = ConfusionMatrixDisplay(result)
    
    if saveConfusionMatrix:
        plt.savefig(testPathName+'/confusion_matrix.png')
    
    if showConfusionMatrix:
        plt.show()

    plt.clf() 
    return disp, accuracy, f1, recall, predictsTotal 


# ### View Performance on Training Set

# In[23]:


def plotTraining(testPathName, testName, history, saveFigure=True, showResult=True):
    plt.style.use('default')
    
    figure, ax = plt.subplots( 1, 2, figsize=(20, 10))
    # plt.suptitle('Accuracy', fontsize=10)
    ax[0].set_title("Loss")
    ax[0].set_ylabel('Loss', fontsize=16)
    ax[0].set_xlabel('Epoch', fontsize=16)
    ax[0].plot(history['train_loss'], label='Training Loss')
    ax[0].plot(history['val_loss'], label='Validation Loss')
    ax[0].legend(loc='upper right')

    ax[1].set_title("Accuracy")
    ax[1].set_ylabel('Accuracy', fontsize=16)
    ax[1].set_xlabel('Epoch', fontsize=16)
    
    ax[1].plot(history['train_acc'], label='Training Accuracy')
    ax[1].plot(history['val_acc'], label='Validation Accuracy')
    ax[1].legend(loc='lower right')

    if saveFigure:
        plt.savefig(testPathName+'/training_history.png')
    
    if showResult:
        plt.show()

    plt.clf() 

    return figure


def plotTrainingPerformances(testPathName, testName, histories, saveFigure=True, showResult=True):
    plt.style.use('default')

    figure, ax = plt.subplots( 2, len(histories), figsize=(80, 20))
    for idx, history in enumerate(histories):
        # plt.suptitle('Accuracy', fontsize=10)
        ax[0][idx].set_title("Loss")
        ax[0][idx].set_ylabel('Loss', fontsize=16)
        ax[0][idx].set_xlabel('Epoch', fontsize=16)
        ax[0][idx].plot(history['train_loss'], label='Training Loss')
        ax[0][idx].plot(history['val_loss'], label='Validation Loss')
        ax[0][idx].legend(loc='upper right')

        ax[1][idx].set_title("Accuracy")
        ax[1][idx].set_ylabel('Accuracy', fontsize=16)
        ax[1][idx].set_xlabel('Epoch', fontsize=16)
        ax[1][idx].plot(history['train_acc'], label='Training Accuracy')
        ax[1][idx].plot(history['val_acc'], label='Validation Accuracy')
        ax[1][idx].legend(loc='lower right')

    plt.suptitle(f'{testName} \nTrainining Performance', fontsize=30)


    if saveFigure:
        plt.savefig(testPathName+'/training_history.png')
    
    if showResult:
        plt.show()

    plt.clf() 

    return figure


# #### Run FullStack of Model

# In[24]:


def plotConfusionMatricies(testPathName, testName, confusion_matricies):
    figure,axis = plt.subplots(1,len(confusion_matricies),figsize=(20, 5))
    for idx in range(len(confusion_matricies)):        
        confusion_matricies[idx].plot(ax=axis[idx])
        confusion_matricies[idx].im_.colorbar.remove()

    figure.suptitle(f'{testName}\nConfusion Matricies')
    plt.savefig(testPathName+'/confusion_matrix.png')
    plt.show()
    plt.clf() 
    
def meanConfidenceInterval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

    min = np.min(a)
    max = np.max(a)

    return [m, h, [min, max], data]

def averagePredictionTotals(predictions, numberOfTrials=5):
    average = {0:0,1:0}
    for prediction in predictions:
        for key, value in prediction.items():
            average[key] += value
    
    for key,value in average.items():
        average[key] = value/numberOfTrials

    return average


def meanConfidenceInterval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

    min = np.min(a)
    max = np.max(a)

    return [m, h, data]

def averageMultilabelMetricScores(scores, numberOfTrials=5, numberOfClasses=2):
    dict = {0:0,1:0}
    averages= [0]*numberOfClasses
    for score in scores:
        for i in range(numberOfClasses):
            averages[i] += score[i]
    averages = [averages[i]/numberOfTrials for i in range(numberOfClasses)]
    
    for key in range(numberOfClasses):
        dict[key] = averages[key]

    mean = np.mean(averages)
    return [mean, dict, scores]

#Runs all model evaluations steps
def runModelFullStack(testPathName, testName, xTrain, yTrain, xVal, yVal, xTest, yTest, modelInformation, trainingTransform=None):
    trainingData, validationData, testingData, training_data_transforms = convertDataToLoaders(xTrain, yTrain, xVal, yVal, xTest, yTest, training_data_transforms = trainingTransform, batchSize=modelInformation['batchSize'])
    model, criterion, scheduler, optimizer = defineModel(dropoutRate=modelInformation['dropoutRate'], learningRate = modelInformation['learningRate'], weight_decay=modelInformation['weight_decay'], model = modelInformation['model'])
    model, criterion, device, history, endingEpoch = trainModel(model, criterion, scheduler, optimizer, trainingData,validationData, numOfEpochs=modelInformation['numOfEpochs'])
    saveResults(testPathName, model, history, training_data_transforms, saveModel=False)
    confusionMatrix, accuracy, f1, recall, predictsTotal = evaluateModelOnTestSet(testPathName, model, testingData, criterion, device, saveConfusionMatrix = True, showConfusionMatrix=False)
    #plotTraining(testPathName, testName, history, saveFigure=True, showResult=True)

    return confusionMatrix, history, accuracy, f1, recall, predictsTotal, endingEpoch+1



# In[43]:


def addEvalDetailToModel(evalDetailLine, dataframe):
    dataframe.loc[dataframe.shape[0]] = [evalDetailLine] + ['']*(len(dataframe.columns)-1)
    
def appendMetricsToXLSX(testPathName, trainingTransform, accuracyInformation, f1Information, recallInformation, predictsTotal, endingEpochs, modelInformation, dataframe):
    rotation = getTransformValue(trainingTransform, desiredTranform=transforms.RandomRotation, desiredTranformValue='degrees')
    elasticTransform = [getTransformValue(trainingTransform, desiredTranform=transforms.ElasticTransform, desiredTranformValue='alpha'), getTransformValue(trainingTransform, desiredTranform=transforms.ElasticTransform, desiredTranformValue='sigma')]
    brightness = getTransformValue(trainingTransform, desiredTranform=transforms.ColorJitter, desiredTranformValue='brightness')
    contrast = getTransformValue(trainingTransform, desiredTranform=transforms.ColorJitter, desiredTranformValue='contrast')
    guassianBlur = [ getTransformValue(trainingTransform, desiredTranform=transforms.GaussianBlur, desiredTranformValue='kernel_size'), getTransformValue(trainingTransform, desiredTranform=transforms.GaussianBlur, desiredTranformValue='sigma')]
    randomHorizontalFlip = getTransformValue(trainingTransform, desiredTranform=transforms.RandomHorizontalFlip, desiredTranformValue='p')
    randomVerticalFlip = getTransformValue(trainingTransform, desiredTranform=transforms.RandomVerticalFlip, desiredTranformValue='p')
    
    exportValue = [testPathName, modelInformation['numOfEpochs'],modelInformation['batchSize'], modelInformation['learningRate'],modelInformation['dropoutRate'],modelInformation['weight_decay'],modelInformation['commandRan'], \
        rotation, elasticTransform, brightness, contrast, guassianBlur, randomHorizontalFlip, randomVerticalFlip, predictsTotal, \
            accuracyInformation[0], f1Information[0], recallInformation[0], \
            accuracyInformation[1], f1Information[1],  recallInformation[1], \
            accuracyInformation[2], f1Information[2], recallInformation[2], \
            endingEpochs]
    
    dataframe.loc[dataframe.shape[0]] = exportValue + ['']*(len(dataframe.columns)-len(exportValue))

def generateKFoldsValidation(identifier,identifierValue, modelInformation, grouped2D, croppedSegmentsList, recistCriteria, cases, k=5,trainingTransform=None):

    # Set the random seed for reproducibility
    random.seed(0)
    torch.manual_seed(0) 
    
    #Keep history of values
    confusion_matricies = []
    histories = []
    accuracies = []
    f1s = []
    recalls = []
    predictionSplits = []
    endingEpochs = []

    print(recistCriteria)
    ## =======================
    ## For undersampling the 0 class
    differenceIn0sTo1s = recistCriteria.count(0) - recistCriteria.count(1)
    print('previous difference', differenceIn0sTo1s)
    indiesToConsiderDropping = []
    for i in range(len(recistCriteria)):
        if recistCriteria[i] == 0:
            indiesToConsiderDropping.append(i)
    
    randomIndicies = random.sample(indiesToConsiderDropping, differenceIn0sTo1s)
    croppedSegmentsList = np.delete(croppedSegmentsList,randomIndicies, axis=0)
    recistCriteria = np.delete(recistCriteria,randomIndicies, axis=0).tolist()
    cases = np.delete(cases,randomIndicies, axis=0).tolist()

    differenceIn0sTo1s = recistCriteria.count(0) - recistCriteria.count(1)
    print('New difference after undersampling', differenceIn0sTo1s)

    ## =======================

    
    if grouped2D: #if >100 then we are doing groupings of 2D images
        stratifiedGroupFolds = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
        stratifiedGroupFolds.get_n_splits(croppedSegmentsList, recistCriteria)
        splits = enumerate(stratifiedGroupFolds.split(croppedSegmentsList, recistCriteria, cases))
        
    else:
        stratifiedFolds = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        stratifiedFolds.get_n_splits(croppedSegmentsList, recistCriteria)
        splits = enumerate(stratifiedFolds.split(croppedSegmentsList, recistCriteria))
        
    for i, (train_index, test_index) in splits:
        
        #Set the name of the test
        testName = f'{identifier}-{identifierValue}'
        
        testPathName = 'Tests/'+testName+f'/foldn{i+1}'
        print(f'{identifier}: foldn{i+1} RUN\n=========================================')
        xTest, yTest = [croppedSegmentsList[i] for i in test_index], [recistCriteria[i] for i in test_index]
        xTrain, yTrain = [croppedSegmentsList[i] for i in train_index], [recistCriteria[i] for i in train_index]
        xVal, yVal = xTest, yTest # Set the validation set to the same as the testing set
        #xVal, yVal = [croppedSegmentsList[i] for i in train_index[:len(xTest)]], [recistCriteria[i] for i in train_index[:len(yTest)]]
        

        # ## Working with Numpy arrays
        xTrain = np.array(xTrain) 
        xTest = np.array(xTest)
        xVal = np.array(xVal)
        yTrain = np.array(yTrain)
        yVal = np.array(yVal)
        yTest = np.array(yTest)

        ## ==============================================================
        # Using SMOTE
        #smote = SMOTE(random_state=42)
        #if len(xTrain.shape)==3:
        #    oneDShape = xTrain[0].shape[0]*xTrain[0].shape[1]
        #    
        #else:
        #    print('xTrain shape',xTrain.shape)
        #    oneDShape = xTrain[0].shape[0]*xTrain[0].shape[1]*xTrain[0].shape[2]

        #singleShape = xTrain[0].shape

        #print('xTrain reshape',xTrain.reshape(xTrain.shape[0],oneDShape).shape)
        #xTrainSmote, yTrain = smote.fit_resample(xTrain.reshape(xTrain.shape[0],oneDShape), yTrain)
        #if len(xTrain.shape)==3:
        #    xTrain = xTrainSmote.reshape(xTrainSmote.shape[0], xTrain[0].shape[0],xTrain[0].shape[1])
        #else:
        #    xTrain = xTrainSmote.reshape(xTrainSmote.shape[0], xTrain[0].shape[0],xTrain[0].shape[1],xTrain[0].shape[2])

        #print('xTrain after Smote', xTrain.shape)
        #print('yTrain after Smote', yTrain.shape)
        #from collections import Counter
        #counter  = Counter(yTest)
        #print('Splits for test Fold',sorted(counter.items()))

        ## ==============================================================
        
        # May or may not need this, def not needed for grouped2D=True
        # xTrain = np.expand_dims(xTrain,axis=-1)
        # xVal = np.expand_dims(xVal,axis=-1)
        # xTest = np.expand_dims(xTest,axis=-1)

        print('xTrain', xTrain.shape)
        print('xVal', xVal.shape)
        print('xTest', xTest.shape)

        ## Get and save results for each fold        
        confusionMatrix, history, accuracy, f1, recall, predictsTotal, endingEpoch = runModelFullStack(testPathName, testName, xTrain, yTrain, xVal, yVal, xTest, yTest, trainingTransform=trainingTransform, modelInformation=modelInformation) 

        confusion_matricies.append(confusionMatrix)
        histories.append(history)
        accuracies.append(accuracy)
        f1s.append(f1)
        recalls.append(recall)
        predictionSplits.append(predictsTotal)
        endingEpochs.append(endingEpoch)
        print('\n\n')

    #assert False
    # Calculate the average of the metrics for the kfolds of this transformation and save it
    kFoldsTestMetrics = {'Prediction averages': averagePredictionTotals(predictionSplits), 'Accuracy':meanConfidenceInterval(accuracies), 'F1 Score':averageMultilabelMetricScores(f1s), 'Recall':averageMultilabelMetricScores(recalls)}
    file = open('Tests/'+testName+'/kFoldsTestMetrics.txt','w')
    for key, value in kFoldsTestMetrics.items():
        file.write(f'{key}: {value}\n')
        print(f'{key}: {value}')
    file.close()

    # Plot training and confusion matrix for each fold as a single .png
    plotConfusionMatricies('Tests/'+testName, testName, confusion_matricies)
    plotTrainingPerformances('Tests/'+testName, testName, histories, saveFigure=True, showResult=True)

    # Append results to the xlsx file
    appendMetricsToXLSX(testPathName, trainingTransform, meanConfidenceInterval(accuracies), averageMultilabelMetricScores(f1s), averageMultilabelMetricScores(recalls), averagePredictionTotals(predictionSplits), endingEpochs, modelInformation, dataframe)


# In[26]:


# # View what the splits are for the models, the test set is {0: 4, 1: 7-8, 2: 6-5}
# stratifiedFolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# stratifiedFolds.get_n_splits(croppedSegmentsList, recistCriteria)
# for i, (train_index, test_index) in enumerate(stratifiedFolds.split(croppedSegmentsList, recistCriteria)):
#         xTrain, xTest = [croppedSegmentsList[i] for i in train_index], [croppedSegmentsList[i] for i in test_index]
#         yTrain, yTest = [recistCriteria[i] for i in train_index], [recistCriteria[i] for i in test_index]

#         # Convert the recist criteria to 0,1,2
#         yTrain = [x-1 for x in yTrain]
#         yTest = [x-1 for x in yTest]

#         #count the categorization splits of the train and test set
#         trainRecistSplit = {y: yTrain.count(y) for y in yTrain}
#         testRecistSplit = {y: yTest.count(y) for y in yTest}

#         # Format the splits nicely into an ordered dictionary
#         myKeys = list(trainRecistSplit.keys())
#         myKeys.sort()
#         trainRecistSplitDisplay = {i: trainRecistSplit[i] for i in myKeys}
#         myKeys = list(testRecistSplit.keys())
#         myKeys.sort()
#         testRecistSplitDisplay = {i: testRecistSplit[i] for i in myKeys}

#         print(f'train recist category split: {trainRecistSplitDisplay}')
#         print(f'test recist category split: {testRecistSplitDisplay}')

#         # ## Working with Numpy arrays
#         xTrain = np.array(xTrain) 
#         xTest = np.array(xTest)
#         yTrain = np.array(yTrain)
#         yTest = np.array(yTest)
#         xTrain =  np.expand_dims(xTrain,axis=-1)
#         xTest =  np.expand_dims(xTest,axis=-1)


# In[44]:


#Make all transforms that I am going to test:
transformsTested = {
    "0":None
    # "20":generateTransform(RandomRotationValue=20, RandomElaticTransform=[20,2], brightnessConstant=20, contrastConstant=20, kernelSize=3, sigmaRange=(0.001,0.4)),
    # "40":generateTransform(RandomRotationValue=40, RandomElaticTransform=[40,4], brightnessConstant=40, contrastConstant=40, kernelSize=3, sigmaRange=(0.001,0.8)),
    # "60":generateTransform(RandomRotationValue=60, RandomElaticTransform=[60,6], brightnessConstant=60, contrastConstant=60, kernelSize=3, sigmaRange=(0.001,1.2)),
    # "80":generateTransform(RandomRotationValue=80, RandomElaticTransform=[80,8], brightnessConstant=80, contrastConstant=80, kernelSize=3, sigmaRange=(0.001,1.6)),
    # "100":generateTransform(RandomRotationValue=100, RandomElaticTransform=[100,10], brightnessConstant=100, contrastConstant=100, kernelSize=3, sigmaRange=(0.001,2.0))    
    #"flips":generateTransform(RandomRotationValue=50, RandomElaticTransform=[100,10], brightnessConstant=100, contrastConstant=100, kernelSize=3, sigmaRange=(0.001,2.0))
}

## Open the dataframe and add the evaluation details
dataframePath='testResults.xlsx'
columns = ['name','numOfEpochs','batchSize','learningRate','dropoutRate','weight_decay', 'commandRan','RandomRotation','ElasticTransform','Brightness','Contrast','GaussianBlur','RandomHorizontalFlip','RandomVerticalFlip','PredictionAverage',
            'AccuracyAverage','F1Average', 'RecallAverage','AccuracySTD','F1STD','RecallSTD','AccuracyData','F1Data','RecallData', 'EndingEpoch']
dataframe = pd.read_excel('testResults.xlsx', header=None, names=columns)
addEvalDetailToModel(evalDetailLine, dataframe)

# Generate the command ran for the test 
commandRan = 'python'
for details in sys.argv:
    print('details',details)
    stringArgs = ['evalDetailLine','modelChosen','votingSystem']
    if details in stringArgs:
        detailArray = details.split('=') 
        details = f'{detailArray[0]}=\'{detailArray[1]}\''
    commandRan += f' {details}'   
print(commandRan)

modelInformation = { 'learningRate': learningRate, 'dropoutRate': dropoutRate, 'batchSize': batchSize, 'numOfEpochs':numOfEpochs, 'weight_decay':weight_decay, 'commandRan': commandRan, 'model': modelChosen, 'votingSystem': votingSystem}
# modelInformation = { 'learningRate': learningRate, 'dropoutRate': dropoutRate, 'batchSize': batchSize, 'numOfEpochs':1, 'weight_decay':weight_decay, 'commandRan': commandRan, 'model': modelChosen, 'votingSystem': votingSystem}


# Run the tests
for key, value in transformsTested.items():
    generateKFoldsValidation(evalDetailLine,"-", croppedSegmentsList=croppedSegmentsList, recistCriteria=recistCriteria, cases=cases, k=5,trainingTransform=value, modelInformation = modelInformation, grouped2D=grouped2D)

dataframe.to_excel(dataframePath, index=False, header=False)


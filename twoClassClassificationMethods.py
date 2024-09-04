#!/usr/bin/env python
# coding: utf-8

# In[12]:


# ! jupyter nbconvert --to python twoClassClassificationMethods.ipynb --output twoClassClassificationMethods.py


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


# In[3]:


def seed_worker(worker_id=42): 
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.Generator().manual_seed(seed)

randomSeed = 42
seed_everything(randomSeed)


# In[4]:


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


# ## Data Augmentation

# In[5]:


# Getting information about the transformations
def generateTransform(RandomHorizontalFlipValue=0.5,RandomVerticalFlipValue=0.5, RandomRotationValue=50, RandomElaticTransform=[0,0], brightnessConstant=0, contrastConstant=0, kernelSize=3, sigmaRange=(0.1,1.0)):
    training_data_transforms = transforms.Compose([
        transforms.RandomRotation(degrees=RandomRotationValue),
        transforms.ElasticTransform(alpha=RandomElaticTransform[0], sigma=RandomElaticTransform[1]),
        transforms.ColorJitter(brightnessConstant, contrastConstant),
        transforms.GaussianBlur(kernel_size = kernelSize, sigma=sigmaRange),
        transforms.RandomHorizontalFlip(p=RandomHorizontalFlipValue),
        transforms.RandomVerticalFlip(p=RandomVerticalFlipValue),
    ]) 
    return training_data_transforms


def getTransformValue(transform, desiredTranform, desiredTranformValue):
    if transform==None or desiredTranform==None or desiredTranformValue==None:
      return None
    for t in transform.transforms:
        if isinstance(t, desiredTranform):
            return t


# In[6]:


## SPLIT THE DATA

def turnDatasetIntoArrays(dataset):
    labels = []
    images = []
    patients = list(dataset.keys())
    for patient in patients:
        labels.append([dataset[patient]['label']])
        images.append([dataset[patient]['images']]) 
    return patients, images, labels

def underSampleData(dataset, trainFolders):
    patients, _, labels = turnDatasetIntoArrays(dataset)
    
    print('Undersample data')
    print('==================================')
    
    indiciesToConsiderDropping = []
    indiciesOfOtherClass = []
    
    for patient in trainFolders:
        print(dataset[patient]['label'])
        if dataset[patient]['label'] == torch.tensor(1, dtype=torch.int64):
            indiciesToConsiderDropping.append(patient)
            print('yes')
        else:
            indiciesOfOtherClass.append(patient)

    differenceIn0sTo1s = len(indiciesToConsiderDropping) - len(indiciesOfOtherClass) 
    print('previous difference', differenceIn0sTo1s)

    randomIndicies = random.sample(indiciesToConsiderDropping, differenceIn0sTo1s)
        
    for patient in randomIndicies:
        del dataset[patient]
        trainFolders.remove(patient)
    
    # Validate the size
    indiciesToConsiderDropping = []
    indiciesOfOtherClass = []
    for patient in trainFolders:
        if dataset[patient]['label'] == torch.tensor(1, dtype=torch.int64):
            indiciesToConsiderDropping.append(patient)
        else:
            indiciesOfOtherClass.append(patient)

    differenceIn0sTo1s = len(indiciesToConsiderDropping) - len(indiciesOfOtherClass) 

    print('New difference after undersampling', differenceIn0sTo1s)
    print('Total size of dataset', len(dataset))
    return dataset, trainFolders

def oversampleData(dataset, trainFolders):
    """Oversamples the training data to make an even number of patients of both labels present in each class"""
    
    patients, _, labels = turnDatasetIntoArrays(dataset)

    print('Oversampling data')
    print('==================================')
    
    indiciesToRepeat = []
    indiciesOfOtherClass = []
    
    for patient in trainFolders:
        if dataset[patient]['label'] == [torch.tensor(0, dtype=torch.int64)]:
            indiciesToRepeat.append(patient)
        else:
            indiciesOfOtherClass.append(patient)

    differenceIn0sTo1s = len(indiciesOfOtherClass) - len(indiciesToRepeat) 
    print('previous difference', differenceIn0sTo1s)

    # Guarentee indicies are repeated if there is room    
    randomIndicies = indiciesToRepeat * (differenceIn0sTo1s // len(indiciesToRepeat))

    # Add on randomly selected indicies to repeat to make up the difference
    randomIndicies = randomIndicies + random.sample(indiciesToRepeat, differenceIn0sTo1s-len(randomIndicies))
    
    # Add the new patients to the dataset and the respective train/test folders
    for patient in indiciesToRepeat:
        i=1
        while True: 
            patientName = f'{patient}_{i}'
            if patientName not in trainFolders:
                patientName = f'{patient}_{i}'
                trainFolders.append(patientName)
                break 
            i+=1
        dataset[patientName] = {'images': dataset[patient]['images'], 'label':dataset[patient]['label']}

    # Validate the size
    indiciesToRepeat = []
    indiciesOfOtherClass = []
    for patient in trainFolders:
        if dataset[patient]['label'] == torch.tensor(0, dtype=torch.int64):
            indiciesToRepeat.append(patient)
        else:
            indiciesOfOtherClass.append(patient)
    differenceIn0sTo1s = len(indiciesOfOtherClass) - len(indiciesToRepeat) 
    
    print('New difference after undersampling', differenceIn0sTo1s)
    print('Total size of dataset', len(dataset))
    return dataset, trainFolders


# In[7]:


# For 2D images:
# ## Working with pytorch tensors
import copy

class PatientData(Dataset):
    def __init__(self, patientsList, allData, grouped2D, segmentsMultiple, transform=None):
        self.patients = patientsList

        # Make the dataset only contain its patients
        self.setData = copy.deepcopy(allData)        
        allPatients = allData.keys()
        for patient in allPatients:
            if patient not in self.patients:
                del self.setData[patient]

        self.transform = transform
        self.grouped2D = grouped2D
        self.segmentsMultiple = segmentsMultiple

    def __len__(self):
        return len(self.patients)* self.segmentsMultiple

    def __getitem__(self, idx):
        if self.grouped2D==False:
            image = self.setData[self.patients[idx]]['images']
            label = self.setData[self.patients[idx]]['label']
        else:
            patient_idx = idx // self.segmentsMultiple
            slice_idx = idx % self.segmentsMultiple

            image = self.setData[self.patients[patient_idx]]['images'][slice_idx]
            label = self.setData[self.patients[patient_idx]]['label']

        # Convert to RGB
        image = Image.fromarray((image * 255).astype(np.uint16))
        image = image.convert("RGB")

        # Apply augmentations if there are any
        if self.transform:
            image = self.transform(image)#.type(torch.float)

        return image, label


def getModelTransformation(model, training_data_transforms=None):
    # Get the baseline tranformations for the model
    if model == 'ResNet50Small2D':
        resizing = transforms.Compose([
            transforms.Resize(224)
        ])
        postProcessing = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
    
    elif model == 'VGG16Small2D':
        resizing = transforms.Compose([
            transforms.Resize(224)
        ])
        postProcessing = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize( mean=[0.48235, 0.45882, 0.40784], std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])
        ]) 
    
    elif model == 'InceptionV3Small2D':
        resizing = transforms.Compose([
            transforms.Resize(299)
        ])
        postProcessing = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 

    elif model == 'XceptionSmall2D':
        resizing = transforms.Compose([
            transforms.Resize(299)
        ])
        postProcessing = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ]) 

    # add in data further data augmentations if specified
    if training_data_transforms == None:
        return transforms.Compose(resizing.transforms + postProcessing.transforms)
    else:
        return transforms.Compose(resizing.transforms + training_data_transforms.transforms + postProcessing.transforms)
    

def convertDataToLoaders(trainPatientList, valPatientList,testPatientList, allData, model, grouped2D, segmentsMultiple, training_data_transforms = None, batchSize=8):
    
    testing_data_transforms = getModelTransformation(model, None)

    # Use the same default training transform as the testing transform if not specified
    if training_data_transforms == None:
        training_data_transforms = testing_data_transforms
    else:
        training_data_transforms = getModelTransformation(model, training_data_transforms)

    # Convert the testing sets to data loaders
    trainingData = PatientData(trainPatientList, allData, grouped2D, segmentsMultiple, transform=training_data_transforms)

    # Need to drop the last layer because of batch normalization 
    trainingData = DataLoader(trainingData, batch_size=batchSize, shuffle=True, worker_init_fn=seed_worker, drop_last=True)#, sampler= TrainBalancedSampler)

    validationData = PatientData(valPatientList, allData, grouped2D, segmentsMultiple, transform=testing_data_transforms)
    validationData = DataLoader(validationData, batch_size=batchSize, shuffle=False, worker_init_fn=seed_worker)

    testingData = PatientData(testPatientList, allData, grouped2D, segmentsMultiple, transform=testing_data_transforms)
    testingData = DataLoader(testingData, batch_size=batchSize, shuffle=False, worker_init_fn=seed_worker)

    return trainingData, validationData, testingData, training_data_transforms


# # Define Models and Training

# In[8]:


class InceptionV3Small2D(torch.nn.Module):
    def __init__(self):
        super(InceptionV3Small2D, self).__init__()

        # inceptionv3 as first layer
        self.model = models.inception_v3(pretrained=True)

        # Modify the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.model.training:
            x = self.model(x)[0]
        else:
            x = self.model(x)
        x = self.sigmoid(x)
        return x 
        
class XceptionSmall2D(torch.nn.Module):
    def __init__(self):
        super(XceptionSmall2D, self).__init__()

        # Resnet50 as first layer
        self.model = timm.create_model('xception', pretrained=True, num_classes=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x 


class VGG16Small2D(torch.nn.Module):
    def __init__(self):
        super(VGG16Small2D, self).__init__()

        # vgg16 as first layer
        self.model = models.vgg16(weights='DEFAULT') 
        
        # Modify the final fully connected layer
        num_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x 
    

class ResNet50Small2D(torch.nn.Module):
    def __init__(self):
        super(ResNet50Small2D, self).__init__()

        #Resnet50 as first layer
        self.model = models.resnet50(weights='IMAGENET1K_V2')

        # Modify the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x 
    
def defineModel(learningRate=0.001, weight_decay=0.01, model = 'ResNet50Small2D'):
    if model == 'ResNet50Small2D':
        model = ResNet50Small2D()
    elif model == 'VGG16Small2D':
        model = VGG16Small2D()
    elif model == 'InceptionV3Small2D':
        model = InceptionV3Small2D()
    elif model == 'XceptionSmall2D':
        model = XceptionSmall2D()

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learningRate, weight_decay=weight_decay)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    return model, criterion, scheduler, optimizer


# In[9]:


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


# In[10]:


def trainModel(model, criterion, scheduler, optimizer, trainingData, validationData, patience=10,numOfEpochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using this device:', device)
    #Send the model to the same device that the tensors are on
    model.to(device)

    earlyStopping = EarlyStopping(patience=patience, minDelta=0)
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
    print('Done Training')
    return model, criterion, device, history, epoch+1


# In[11]:


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



# In[12]:


## SAVE CONTENTS
def saveResults(testPathName, model, history, training_data_transforms, saveModel=True):
    os.makedirs(testPathName, exist_ok=True)

    #Save history as pickle
    with open(testPathName+'/history.pkl', 'wb') as fp:
        pkl.dump(history, fp)

    # Save weigths of model
    if saveModel:
        torch.save(model.state_dict(), testPathName+'/model.pt')

    # Save transformations for easy access
    f = open(testPathName + '/training_data_transforms.txt', 'w')
            
    for line in training_data_transforms.__str__():
        f.write(line)
    f.close()


# In[13]:


def writeDictionaryToTxtFile(filePath,dictionary, printLine=False):
    f = open(filePath, 'w')
    for key, value in dictionary.items():
        f.write(f'{key}: {value}\n')
        if printLine:
            print(f'{key}: {value}')
    f.close()


# In[14]:


## EVALUATE PERFORMANCE ON TESTING SET
def formatDataFromGroupVoting(outputs):
    ## Outputs are in the format of [array([[False],[False],[False],[False],[False],[False], ....]), array([[False], .....)]
    return [pred.tolist()[0] for sublist in outputs for pred in sublist]

def evaluateGroupVoting(model, loader, criterion, device, votingSystem, segmentsMultiple):
    model.eval()
    predictions = []
    probabilities = []
    correctLabelsTemp = []
    correctLabels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1, 1)

            labels = labels > 0.5

            outputs = model(inputs)
            predicted = outputs > 0.5

            probabilities.append(outputs.cpu().numpy())
            predictions.append(predicted.cpu().numpy())
            correctLabelsTemp.append(labels.cpu().numpy())
    
    # Ignores grouped voting
    probabilities = formatDataFromGroupVoting(probabilities)
    predictions = formatDataFromGroupVoting(predictions)
    correctLabelsTemp = formatDataFromGroupVoting(correctLabelsTemp)

    if votingSystem=='singleLargest' or segmentsMultiple==1:
        return probabilities, predictions, correctLabelsTemp

    ## Grouping predictions by patient
    ## ==============================================================
    # all_probs = np.concatenate(probabilities, axis=0)

    # Grouping predictions by patient
    patient_probs = [] # The confidence of the model
    patient_labels = [] # the label given by the model. >=0.5 is 1, <0.5 is 0
     
    
    ## Get the classification based from the patient based on ...
    for i in range(0, len(probabilities), segmentsMultiple):
        # Get probabilties and labels for the patient
        patient_prob = probabilities[i:i + segmentsMultiple]
        patient_pred_labels = [pred>=0.5 for pred in patient_prob]

        correctLabel = correctLabelsTemp[i]
        correctLabels.append(correctLabel)

        ## Average all confidences to get the highest probability label, used as a tie breaker
        patient_prob = np.mean(patient_prob) 
        prob_label_max = patient_prob >= 0.5 
        
        ## MAJOURITY VOTING
        ##==============================================================
        # Get counts for the labels 
        if votingSystem=='majority':
            predictedLabels = {True:0,False:0}

            uniqueLabels, label_counts = np.unique(patient_pred_labels, return_counts=True)

            for i in range(len(uniqueLabels)):
                predictedLabels[uniqueLabels[i]] = label_counts[i]

            majorityVote = True if predictedLabels[True] > predictedLabels[False] else False

            #Tie breaker based on the highest probability
            if predictedLabels[False] == predictedLabels[True]:
                majorityVote = prob_label_max
            patient_probs.append(patient_prob)
            patient_labels.append(majorityVote)
        ##==============================================================
        ## AVERAGE
        ##==============================================================
        elif votingSystem=='average':
            patient_probs.append(patient_prob)
            patient_labels.append(prob_label_max)

    return patient_probs, patient_labels, correctLabels


# In[15]:


def evaluateModelOnTestSet(testPathName, model, testingData, criterion, device, votingSystem, segmentsMultiple, saveConfusionMatrix = True, showConfusionMatrix=True, showROCCurve=True, saveROCCurve=True):
    predictProbs, predictions, ans = evaluateGroupVoting(model, testingData, criterion, device, votingSystem, segmentsMultiple)
    
    predictsTotal = dict(zip([0,1],[predictions.count(False),predictions.count(True)]))
    ansTotal = dict(zip([0,1],[ans.count(False),ans.count(True)]))

    print('ans length',len(ans))

    # Test metrics
    print('---------------------------------------\nTesting Metrics')
    
    accuracy = metrics.accuracy_score(ans, predictions)
    f1 = list(metrics.f1_score(ans, predictions, average=None))  
    recall = list(metrics.recall_score(ans, predictions, average=None))  
    precision = list(metrics.precision_score(ans, predictions, average=None))
    
    fpr, tpr, thresholds = metrics.roc_curve(ans, predictProbs, pos_label=1)
    roc_auc = metrics.auc(fpr,tpr)
    rocCurveDisplay = metrics.RocCurveDisplay(fpr=fpr,tpr=tpr,roc_auc=roc_auc, pos_label=1)

    
    testingMetrics = {'PredictionSplits': predictsTotal, 'AnswerSplits': ansTotal, 'Predictions': [1 if ans else 0 for ans in predictions], 'Answers    ': [ 1 if ans else 0 for ans in ans],  
                      'Accuracy':accuracy, 'F1':f1, 'Recall':recall, 'Precision':precision, 'ROC-AUC': roc_auc, 'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}

    if showROCCurve:
        plt.close()
        rocCurveDisplay.plot()
    
    if saveROCCurve:
        plt.savefig(testPathName+'/rocCurve.png')

    file = open(testPathName+'/testingMetrics.txt','w')
    for key, value in testingMetrics.items():
        file.write(f'{key}: {value}\n')
    file.close()


    for key, value in testingMetrics.items():
        print(f'{key}: {value}')


    print('---------------------------------------\nConfusion Matrix:')
    # Confusion Matrix
    confusionMatrixResult = confusion_matrix(ans,predictions,normalize='pred')
    
    confusionMatrixDisp = ConfusionMatrixDisplay(confusionMatrixResult)
    if showConfusionMatrix:
        plt.close()
        confusionMatrixDisp.plot()

    if saveConfusionMatrix:
        plt.savefig(testPathName+'/confusion_matrix.png')
    
    return confusionMatrixDisp, rocCurveDisplay, testingMetrics


# In[16]:


## PLOT TRAINING AND CONFUSION MATRICIES
def plotTraining(testPathName, testName, history, saveFigure=True, showResult=True):
    plt.style.use('default')
    
    figure, ax = plt.subplots( 1, 2, figsize=(10, 5))
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
        plt.close()
        plt.show()

    return figure



def plotTrainingPerformances(testPathName, testName, histories, saveFigure=True, showResult=True):
    plt.style.use('default')

    figure, ax = plt.subplots( 2, len(histories), figsize=(40, 20))
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
        plt.savefig(testPathName+'/training_histories.png', format='png')
    
    if showResult:
        plt.close()
        plt.show()

    return figure

def plotConfusionMatricies(testPathName, testName, confusion_matricies, showMatricies=True):
    figure,axis = plt.subplots(1,len(confusion_matricies),figsize=(30, 5))
    for idx in range(len(confusion_matricies)):        
        confusion_matricies[idx].plot(ax=axis[idx])
        confusion_matricies[idx].im_.colorbar.remove()

    figure.suptitle(f'{testName}\nConfusion Matricies')
    plt.savefig(testPathName+'/confusion_matrix.png')
    if showMatricies:
        plt.close()
        plt.show()

def plotROCCurves(testPathName, testName, rocCurves, showMatricies=True):
    figure,axis = plt.subplots(1,len(rocCurves),figsize=(30, 5))
    for idx in range(len(rocCurves)):        
        rocCurves[idx].plot(ax=axis[idx])
        #confusion_matricies[idx].im_.colorbar.remove()

    figure.suptitle(f'{testName}\nROC-AUC curves')
    plt.savefig(testPathName+'/ROC-AUC.png')
    if showMatricies:
        plt.close()
        plt.show()


# In[4]:


## AVERAGES CALCULATIONS

def formatValues(value, significantDigits=4):
    return round(value,significantDigits)

def meanConfidenceInterval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

    min = np.min(a)
    max = np.max(a)
    return [formatValues(m), formatValues(h), [formatValues(val) for val in data]]

def averagePredictionTotals(predictions, numberOfTrials=5):
    average = {0:0,1:0}
    for prediction in predictions:
        for key, value in prediction.items():
            average[key] += value
    
    for key,value in average.items():
        average[key] = formatValues(value/numberOfTrials)

    return average

def averageMultilabelMetricScores(scores, numberOfTrials=5, numberOfClasses=2):
    dict = {0:0,1:0}
    averages= [0]*numberOfClasses
    for score in scores:
        for i in range(numberOfClasses):
            averages[i] += score[i]
    averages = [averages[i]/numberOfTrials for i in range(numberOfClasses)]
    
    for key in range(numberOfClasses):
        dict[key] = formatValues(averages[key])

    
    singleScoreAverages = meanConfidenceInterval(np.mean(scores, axis=1))

    return [singleScoreAverages[0], singleScoreAverages[1], dict, [ [formatValues(val) for val in score] for score in scores]]


# In[18]:


## APPEND RESULTS TO XLSX
def addEvalDetailToModel(evalDetailLine, dataframe):
    exportValue = [evalDetailLine]
    dataframe.loc[dataframe.shape[0]] = exportValue + ['']* (len(dataframe.columns) - len(exportValue))

def appendMetricsToXLSX(evalDetailLine, testName, kFoldsTestMetrics, dataframe):

    predictionSplits = f"{kFoldsTestMetrics['PredictionSplits']}"
    average = f"{kFoldsTestMetrics['Accuracy'][0]} ± {kFoldsTestMetrics['Accuracy'][1]}"
    f1 = f"{kFoldsTestMetrics['F1'][0]} ± {kFoldsTestMetrics['F1'][1]} , {kFoldsTestMetrics['F1'][2]}"
    recall = f"{kFoldsTestMetrics['Recall'][0]} ± {kFoldsTestMetrics['Recall'][1]} , {kFoldsTestMetrics['Recall'][2]}"
    precision = f"{kFoldsTestMetrics['Precision'][0]} ± {kFoldsTestMetrics['Precision'][1]} , {kFoldsTestMetrics['Precision'][2]}"
    rocAuc = f"{kFoldsTestMetrics['ROC-AUC'][0]} ± {kFoldsTestMetrics['ROC-AUC'][1]}"
    endingEpochs = f"{kFoldsTestMetrics['endingEpochs']}"
    accuracyData = f"{kFoldsTestMetrics['Accuracy'][2]}"
    f1Data = f"{kFoldsTestMetrics['F1'][3]}"
    recallData = f"{kFoldsTestMetrics['Recall'][3]}"
    precisionData = f"{kFoldsTestMetrics['Precision'][3]}"
    rocAUCData = f"{kFoldsTestMetrics['ROC-AUC'][2]}"
    
    exportValue = [evalDetailLine, testName, predictionSplits, average, f1, recall, precision, rocAuc, endingEpochs, accuracyData, f1Data, recallData, precisionData, rocAUCData]
  
    dataframe.loc[dataframe.shape[0]] = exportValue 


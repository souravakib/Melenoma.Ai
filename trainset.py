# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 06:22:06 2023

@author: ssour
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
from tqdm.notebook import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, random_split, SubsetRandomSampler, ConcatDataset

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

class CustomTransform:
    def __init__(self, probability=0.3, rotation=180, resize=(227, 227)):
        self.probability = probability
        self.rotation = rotation
        self.resize = resize
        self.transforms_augment = transforms.Compose([transforms.RandomRotation(self.rotation),
                                                      transforms.RandomResizedCrop(self.resize)])
        self.transforms_standard = transforms.Compose([transforms.Resize(self.resize),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.485, 0.456, 0.406), 
                                                                            (0.229, 0.224, 0.225))])

    def __call__(self, img):
        if random.random() < self.probability:
            img = self.transforms_augment(img)
        return self.transforms_standard(img)


transform = CustomTransform()

batch_size = 64

train_set = torchvision.datasets.ImageFolder('C:/E_Drive/kaBHOOM/UPM/FYP/Melenoma/archive (1)/melanoma_cancer_dataset/train', 
                                             transform=transform)

test_set = torchvision.datasets.ImageFolder('C:/E_Drive/kaBHOOM/UPM/FYP/Melenoma/archive (1)/melanoma_cancer_dataset/test', 
                                             transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
                                           shuffle=True, num_workers=2)

classes = train_set.classes
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.AdaptiveAvgPool2d((6, 6))
                                     )
        
        self.classifier = nn.Sequential(nn.Linear(256 * 6 * 6, 4096),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(4096, 4096),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(4096, 2)
                                       )


    def forward(self, x):
        x = self.features(x)
        
        x = x.view(-1, 6 * 6 * 256)
        
        x = self.classifier(x)
        return x


model = AlexNet()
model.to(device)

criterion = nn.CrossEntropyLoss()

torch.manual_seed(42)

#dataset = ConcatDataset([train_set, test_set])
dataset = train_set

num_epochs = 30

k = 10

splits = KFold(n_splits = k, shuffle = True, random_state = 42)

foldperf = {}


def train_epoch(model, device, dataloader, loss_fn, optimizer):
    train_loss, train_correct = 0.0, 0
    model.train()
    for images, labels in dataloader:

        images,labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()

    return train_loss, train_correct
  
def valid_epoch(model, device, dataloader, loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    for images, labels in dataloader:

        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = loss_fn(output,labels)
        valid_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data,1)
        val_correct += (predictions == labels).sum().item()

    return valid_loss, val_correct

for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

    print('Fold {}'.format(fold + 1))

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size = batch_size, sampler = train_sampler)
    test_loader = DataLoader(dataset, batch_size = batch_size, sampler = test_sampler)
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}

    for epoch in tqdm(range(num_epochs)):
        train_loss, train_correct=train_epoch(model, device, train_loader, criterion, optimizer)
        test_loss, test_correct=valid_epoch(model, device, test_loader, criterion)

        train_loss = train_loss / len(train_loader.sampler)
        train_acc = train_correct / len(train_loader.sampler) * 100
        test_loss = test_loss / len(test_loader.sampler)
        test_acc = test_correct / len(test_loader.sampler) * 100

        print("Epoch:{}/{} || AVG Training Loss:{:.3f} || AVG Test Loss:{:.3f} || AVG Training Acc {:.2f} % || AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                                                             num_epochs,
                                                                                                             train_loss,
                                                                                                             test_loss,
                                                                                                             train_acc,
                                                                                                             test_acc))
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

    foldperf['fold{}'.format(fold+1)] = history
    
torch.save(model,'melanoma_CNN.pt') 





nb_classes = 2

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, 
                                          shuffle=True, num_workers=2)

confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    test_running_corrects = 0.0
    test_total = 0.0
    model.eval()
    for i, (test_inputs, test_labels) in enumerate(test_loader, 0):
        test_inputs, test_labels = test_inputs.cuda(), test_labels.cuda()

        test_outputs = model(test_inputs)
        _, test_outputs = torch.max(test_outputs, 1)
        
        test_total += test_labels.size(0)
        test_running_corrects += (test_outputs == test_labels).sum().item()
        
        for t, p in zip(test_labels.view(-1), test_outputs.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        
        
    print(f'Testing Accuracy: {(100 * test_running_corrects / test_total)}%')
print(f'Confusion Matrix:\n {confusion_matrix}')









precision = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
recall = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])

f1_score = 2 * precision * recall / (precision + recall)
print(f'F1 Score: {f1_score: .3f}')
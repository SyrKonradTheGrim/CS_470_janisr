# Program written to train two CNN models of different variations
# Written by Ryan Janis
# Dec 2023 --- Python 3.10.11

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as v2
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

class TheCNN(nn.Module):
    # Class written to serve as the first of two CNN models that is much more complex than the second
    # Written by Ryan Janis
    # Dec 2023 --- Python 3.10.11
    
    def __init__(self, class_cnt):
        super(TheCNN, self).__init__()
        self.net_stack = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding = "same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding = "same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, 3, padding = "same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding = "same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            nn.Conv2d(128, 256, 3, padding = "same"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding = "same"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            nn.Flatten(),
            
            nn.Linear(256 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, class_cnt)
        )

    def forward(self, x):        
        logits = self.net_stack(x)
        return logits

class AnotherCNN(nn.Module):
    # Class written to serve as the second of two CNN models which is much simpler than the first
    # Written by Ryan Janis
    # Dec 2023 --- Python 3.10.11
    
    def __init__(self, class_cnt):
        super(AnotherCNN, self).__init__()
        self.net_stack = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding = "same"),
            nn.ReLU(),
            nn.Conv2d(32,32, 3, padding = "same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding = "same"),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding = "same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            
            nn.Linear(4096, 32),
            nn.ReLU(),
            nn.Linear(32, class_cnt)
        )
        

    def forward(self, x):        
        logits = self.net_stack(x)
        return logits
    
################################################################################################################################################
# These next couple functions are very self explanatory and it would be kinda silly to do the whole shabang, but rest assured all of them are: #
# Written by Ryan Janis                                                                                                                        #
# Dec 2023 --- Python 3.10.11                                                                                                                  #
################################################################################################################################################

def get_approach_names():
    return ["TheCNN", "AnotherCNN"]

def get_approach_description(approach_name):
    if approach_name == "TheCNN":
        return "A CNN with three convolutional layers and three fully-connected layers, utilizing dropout and batch normalization."
    elif approach_name == "AnotherCNN":
        return "A simpler CNN with two convolutional layers and two fully connected layers."

def get_data_transform(approach_name, training):
    if training:
        # Data augmentation for training data
        data_transform = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(10),
            v2.RandomResizedCrop(32, scale = (0.8, 1.0)),
            v2.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.2),
            v2.ToTensor(),
            v2.ConvertImageDtype(torch.float32),
        ])
    else:
        # No data augmentation for non-training data
        data_transform = v2.Compose([
            v2.ToTensor(),
            v2.ConvertImageDtype(torch.float32),
        ])

    return data_transform

def get_batch_size(approach_name):
    return 64  

def create_model(approach_name, class_cnt):
    if approach_name == "TheCNN":
        return TheCNN(class_cnt)
    elif approach_name == "AnotherCNN":
        return AnotherCNN(class_cnt)

def train_model(approach_name, model, device, train_dataloader, test_dataloader):
    # Function written to train the created models
    # Written by Ryan Janis
    # Dec 2023 --- Python 3.10.11
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    # Training loop
    for epoch in range(5):  # Number of epochs can be changed here!
        model.train()
        for inputs, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{5}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Print test set accuracy after each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_dataloader, desc=f"Epoch {epoch + 1}/{5} - Testing"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch + 1}/{5}, Test Accuracy: {100 * correct / total}%')

    return model

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models
import torchvision.transforms as transforms
from torch.utils.data import Dataset as Dataset
import matplotlib.pyplot as plt
import time
import os
import copy
import dlib
import pandas as pd
import torchvision.models as models
from skimage import io, transform
from sklearn import preprocessing

import sys
sys.path.append('/usr/local/lib/python3.6/dist-packages')

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)


class FaceImageDataset(Dataset):

    def __init__(self, csv_path, rootdir, encoder, transform=None):
        self.csv_file = pd.read_csv(csv_path)
        self.rootdir = rootdir
        self.transform = transform
        self.encoder = encoder

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        img_name = os.path.join(self.rootdir, self.csv_file.iloc[index, 0])
        image = io.imread(img_name)

        age = self.csv_file.iloc[index, 1]
        gender = self.csv_file.iloc[index, 2]
        race = self.csv_file.iloc[index, 3]
        service_test = self.csv_file.iloc[index, 4]

        if(self.transform):
            image = self.transform(image)

        age_list = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', 'more than 70']
        race_list = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
        gender_list = ['Male', 'Female']

        service_test = 1 if service_test == True else 0

        labels = [age_list.index(age), gender_list.index(gender), race_list.index(race), service_test]

        label_tensor = torch.as_tensor(labels)

        sample = [image, label_tensor]

        return sample

if __name__ == "__main__":
    #Please create a csv with one column 'img_path', contains the full paths of all images to be analyzed.
    #Also please change working directory to this file.
    dlib.DLIB_USE_CUDA = True
    print("using CUDA?: %s" % dlib.DLIB_USE_CUDA)
    encoder = preprocessing.LabelEncoder()
    #Run training & validation
    #format: file,age,gender,race,service_test
    train_data = FaceImageDataset(csv_path='./fairface_label_train.csv',
                                    rootdir='./',
                                    encoder = encoder,
                                    transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((224, 224)),
                                        transforms.RandomCrop(224),
                                        transforms.ToTensor()
                                    ]))
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=50,
                                            shuffle=True, num_workers=0)
    val_data = FaceImageDataset(csv_path='./fairface_label_val.csv',
                                    rootdir='./',
                                    encoder = encoder,
                                    transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((224, 224)),
                                        transforms.RandomCrop(224),
                                        transforms.ToTensor()
                                    ]))
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=50,
                                            shuffle=True, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #print(train_data.__getitem__(0))

    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
        
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, 4))
    criterion = nn.MultiLabelMarginLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    model.to(device)

    '''predefined code'''

    epochs = 1
    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        for inputs, labels in train_dataloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in val_dataloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()
                        '''
                        ps = torch.exp(logps)
                        print(ps)
                        top_p, top_class = ps.topk(1, dim=0)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        '''
                        outputs = torch.sigmoid(logps)
                        outputs[outputs >= 0.5] = 1
                        accuracy += (outputs == labels).sum()
                train_losses.append(running_loss/len(train_dataloader))
                val_losses.append(test_loss/len(val_dataloader))                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(val_dataloader):.3f}.. "
                    f"Test accuracy: {accuracy/len(val_dataloader):.3f}")
                running_loss = 0
                model.train()
    torch.save(model, 'fairface_res.pth')
    
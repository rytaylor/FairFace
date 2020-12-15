from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset as Dataset
import matplotlib.pyplot as plt
import time
import os
import copy
import dlib
import pandas as pd
import torchvision.models as models
from skimage import io, transform

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

    def __init__(self, csv_path, rootdir, transform=None):
        self.csv_file = pd.read_csv(csv_path)
        self.rootdir = rootdir
        self.transform = transform

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

        sample = {'image': image, 'age': age, 'gender': gender, 'race': race, 'service_test': service_test}

        if(self.transform):
            sample = self.transform(sample)

        return sample

if __name__ == "__main__":
    #Please create a csv with one column 'img_path', contains the full paths of all images to be analyzed.
    #Also please change working directory to this file.
    dlib.DLIB_USE_CUDA = True
    print("using CUDA?: %s" % dlib.DLIB_USE_CUDA)
    #Run training & validation

    train_paths = FaceImageDataset(csv_path='./fairface_label_train.csv',
                                    rootdir='/train')

    #format: file,age,gender,race,service_test

    print(train_paths.__getitem__(0))

    '''
    
    resnet18 = models.resnet18()
    loss_func = nn.MSELoss(reduction='sum')
    learning_rate = 0.0001

    resnet18()

    print(resnet18)
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
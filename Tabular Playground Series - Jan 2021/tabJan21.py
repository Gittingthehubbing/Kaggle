# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 19:38:46 2021

@author: Quiet

Tabular Playground Series - Jan 2021
"""
import torch as t
from torch.utils.data.dataloader import DataLoader as dl
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data  import random_split #for train test split
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.linear_model import BayesianRidge as bayR
from sklearn.preprocessing import StandardScaler as ss
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as ss
import os

class net(nn.Module):
    
    def __init__(self, x, y):
        super().__init__()
        self.xShape = x.shape
        self.yShape = y.shape
        self.model = nn.Sequential(
            nn.Linear(self.xShape[1],128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,1))
    
    def forward(self,x):
        return self.model(x)
    
mainDir = r"E:\KaggleData\Tabular Playground Series - Jan 2021"

os.chdir(mainDir)

trainDataRaw = pd.read_csv("train.csv")
testDataRaw = pd.read_csv("test.csv")

targetCol = "target"
idCol = "id"

trD = trainDataRaw.drop([targetCol, idCol],axis=1)
teD = testDataRaw.drop([idCol],axis=1)
trTarget = trainDataRaw[targetCol]

xTr, xE, yTr, yE = tts(trD, trTarget)

# Normalise
ssTr = ss().fit(xTr)

xTrT = ssTr.transform(xTr)
xET = ssTr.transform(xE)
teDT = ssTr.transform(teD)

ssTarget = ss().fit(yTr.to_numpy().reshape(-1 ,1))
yTrT = ssTarget.transform(yTr.to_numpy().reshape(-1 ,1))
yET = ssTarget.transform(yE.to_numpy().reshape(-1 ,1))

#make Dataloaders
trDl = dl(TensorDataset(t.Tensor(xTrT), t.Tensor(yTrT)))
evalDl = dl(TensorDataset(t.Tensor(xET), t.Tensor(yET)))

# Shallow tests



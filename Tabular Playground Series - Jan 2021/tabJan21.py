# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 19:38:46 2021

@author: Quiet

Tabular Playground Series - Jan 2021
"""
import matplotlib.pyplot as plt
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
from sklearn.metrics import mean_squared_error as mse
import os
import datetime

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

epochs = 50
lr = 3e-4
batchSize = 512

logTB = True

doShallows =False

os.chdir(mainDir)

if logTB:
    dateTimeNow =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writerTr = SummaryWriter("logs/"+dateTimeNow+"TrainLosses")
    writerEval = SummaryWriter("logs/"+dateTimeNow+"EvalLosses")
    
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
teDT = ssTr.transform(teD).reshape(-1 ,1)

ssTarget = ss().fit(yTr.to_numpy().reshape(-1 ,1))
yTrT = ssTarget.transform(yTr.to_numpy().reshape(-1 ,1))
yET = ssTarget.transform(yE.to_numpy().reshape(-1 ,1))

#make Dataloaders
trDl = dl(TensorDataset(t.Tensor(xTrT), t.Tensor(yTrT)),batch_size=batchSize)
evalDl = dl(TensorDataset(t.Tensor(xET), t.Tensor(yET)),batch_size=batchSize)

# Shallow tests

if doShallows:
    ranForest = rfr(verbose=1) # mse: 0.130517
    bay = bayR(verbose=1)
    svr = SVR(verbose=1)
    
    bay.fit(xTrT, yTrT)
    bay = mse(bay.predict(xTrT),yTrT)
    print('Forest MSE: ',bay)


# nn approach
device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

nnNet = net(xTrT, yTrT).to(device)
opti = t.optim.Adam(nnNet.parameters(), lr = lr)
criterion = nn.MSELoss()

trLosses =[]
evalLosses = []
for e in tqdm(range(epochs)):
    epochLossTr = []
    epochLossE = []
    for i, (x,y) in enumerate(trDl):
        nnNet.train()
        out = nnNet(x.to(device))
        lossTr = criterion(y.to(device), out)
        opti.zero_grad()
        lossTr.backward()
        opti.step()
        epochLossTr.append(lossTr.cpu().detach().numpy().item())
        writerTr.add_scalar("lossTr", lossTr.cpu().detach().numpy().item(),i+e)
        
        
        if i%50==0:
            with t.no_grad():
                for iTe,(xTe, yTe) in enumerate(evalDl):
                    nnNet.eval()
                    outTe = nnNet(xTe.to(device))
                    lossTe = criterion(yTe.to(device),outTe)
                    epochLossE.append(lossTe.cpu().detach().numpy().item())
                    writerEval.add_scalar("lossEval", lossTe.cpu().detach().numpy().item(),iTe+e)
    trLosses.append(np.mean(epochLossTr))
    evalLosses.append(np.mean(epochLossE))
    
# plot losses

plt.plot(trLosses, 'k.')
plt.plot(evalLosses, 'r.')
plt.show()
plt.close()
                    
        



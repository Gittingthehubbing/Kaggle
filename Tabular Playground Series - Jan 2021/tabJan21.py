# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 19:38:46 2021

@author: Quiet

Tabular Playground Series - Jan 2021
"""
import matplotlib.pyplot as plt
import torch as t
from torch.utils.data.dataloader import DataLoader as dl
from torch.utils.data import TensorDataset, Dataset, random_split, Subset
# from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.functional as F
import pytorch_lightning as pl
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
from tensorboardX import SummaryWriter

import pytorch_lightning as pl


class LitModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(14, 1)

    def forward(self, x):
        return t.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
       return t.optim.Adam(self.parameters(), lr=0.02)

class lnet(pl.LightningModule):

    def __init__(self, x, y):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(14,128),
            nn.ReLU(),
            nn.Linear(128,1))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # x = x.view(x.size(0),-1)
        z = self(x)
        loss = F.mse_loss(y, z)
        # self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = t.optim.Adam(self.parameters(), 
                                 lr=lr)
        # #,
        #                          momentum=0.9,
        #                          weight_decay=5e-4
        return optimizer

    # def validation_step(self, batch, batch_idx):
    #     x, y, = batch
    #     out = self(x)
    #     val_loss  = F.cross_entropy(out, y)
    #     return val_loss

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     out = self(x)
    #     loss = F.cross_entropy(out, y)
    #     return loss

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

epochs = 20
lr = 3e-4
batchSize = 1024
addL1Reg = True

logTB = 1

doShallows =False
doPYNN = 1
doLightning = 0

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

if doPYNN:
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
            if addL1Reg:
                    l2_reg = t.tensor(0.).to(device)
                    for parameter in nnNet.parameters():
                        l2_reg += t.linalg.norm(parameter)
                    lossTr += 0.001 * l2_reg
            opti.zero_grad()
            lossTr.backward()
            opti.step()
            epochLossTr.append(lossTr.cpu().detach().numpy().item())
            if logTB:
                writerTr.add_scalar("lossTr", lossTr.cpu().detach().numpy().item(),i+e)
    
    
            if i%50==0:
                with t.no_grad():
                    for iTe,(xTe, yTe) in enumerate(evalDl):
                        nnNet.eval()
                        outTe = nnNet(xTe.to(device))
                        lossTe = criterion(yTe.to(device),outTe)
                        if addL1Reg:
                            l2_reg = t.tensor(0.).to(device)
                            for parameter in nnNet.parameters():
                                l2_reg += t.linalg.norm(parameter)
                            lossTe += 0.001 * l2_reg
                        epochLossE.append(lossTe.cpu().detach().numpy().item())
                        if logTB:
                            writerEval.add_scalar("lossEval", lossTe.cpu().detach().numpy().item(),iTe+e)
        trLosses.append(np.mean(epochLossTr))
        evalLosses.append(np.mean(epochLossE))
    
    # plot losses
    
    plt.plot(trLosses, 'k.')
    plt.plot(evalLosses, 'r.')
    plt.show()
    plt.close()
    
    nnNet.eval()
    submissionY = nnNet(t.Tensor(teDT).view(-1,1).to(device))
    dfSubmission = pd.DataFrame(data=submissionY,
                                index=np.arange(0,len(submissionY)))

## nn with LightningModule
if doLightning:
    trDl = dl(TensorDataset(t.Tensor(xTrT), t.Tensor(yTrT)))

    trainer = pl.Trainer(gpus=1,max_epochs=epochs)
    trainer.fit(LitModel, trDl)
    # trainer.fit(lnet, trDl)

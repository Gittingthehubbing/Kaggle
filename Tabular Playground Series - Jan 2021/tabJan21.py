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
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.linear_model import BayesianRidge as bayR
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor as gradB
from sklearn.preprocessing import StandardScaler as ss
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as ss
from sklearn.metrics import mean_squared_error as mse
from joblib import parallel_backend # to improve sklearn training speed
import os
import datetime
from tensorboardX import SummaryWriter
from sklearn.decomposition import PCA
import pytorch_lightning as pl
import optuna as opt

#TODO implement optuna for shallows
#TODO Rebalance data (Might not be needed)
#TODO increase NN complexity


class LitModel(pl.LightningModule):

    def __init__(self,x):
        super().__init__()
        self.lr = lr
        self.l1 = nn.Linear(x.shape[1], 1)

    def forward(self, x):
        return t.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = t.sqrt(F.mse_loss(y_hat, y))
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
       return t.optim.Adam(self.parameters(), lr=(self.lr or self.learning_rate))




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
        loss = t.sqrt(F.mse_loss(y, z))
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


class dset(Dataset):


    def __init__(self, xAll,yAll):
        super().__init__()
        self.xAll = xAll
        self.yAll = yAll

    def __len__(self):
        return self.xAll.shape[0]

    def __getitem__(self,i):
        x = self.xAll[i]
        y = self.xAll[i]
        return (t.Tensor(x,dtype=t.float), t.Tensor(y,dtype=t.float))


def makeModel(trial):
    # sets layers and number of neurons for each trial
    #builds dict with parameters that can be used to make best model
    nl = trial.suggest_int("nL",minLayers,maxLayers)
    layers=[]
    inFeat = xTrT.shape[1] #assumes 1D data for each sample
    lastOut = yTrT.shape[1]

    for i in range(nl):
        outFeat = trial.suggest_int(f"n{i}",minNeurons,maxNeurons)
        layers.append(nn.Linear(inFeat, outFeat))
        actiLayer = trial.suggest_categorical(f"a{i}",possibleActiFuncs)
        layers.append(getattr(nn, actiLayer)())
        dropOutRatio = trial.suggest_float(f"dropoutL{i}",minDropOut,maxDropOut)
        layers.append(nn.Dropout(dropOutRatio))
        inFeat = outFeat
    layers.append(nn.Linear(inFeat, lastOut))

    return nn.Sequential(*layers)

def shallowObjective(trial):

    modelname = trial.suggest_categorical("modelname",possibleShallows)
    idxArr = np.arange(len(xTrT))
    np.random.shuffle(idxArr)
    idxSel = idxArr[:numShallowSamples]
    if modelname == "SVR":
        C = trial.suggest_float("C",1e0,1e4,log=True)
        kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
        if kernel == 'poly':
            polyDeg = trial.suggest_int("polyDeg",2,6)
        else:
            polyDeg = 3
        if kernel in ['rbf','poly','sigmoid']:
            gamma = trial.suggest_categorical('gamma',['scale', 'auto'])
        else:
            gamma = 'scale'
        if kernel in ['poly','sigmoid']:
            coef0 = trial.suggest_float('coef0',0.,0.2)
        else:
            coef0 = 0.
        model = SVR(kernel=kernel, degree=polyDeg, gamma=gamma, coef0=coef0, C=C,verbose=1)
        with parallel_backend('threading', n_jobs=-1):
            model.fit(xTrT[idxSel], yTrT[idxSel].reshape(-1))
            trainLoss = mse(yTrT[idxSel].reshape(-1),model.predict(xTrT[idxSel]).reshape(-1))
            evalLoss = mse(yET.reshape(-1),model.predict(xET).reshape(-1))
            print(modelname,f' TrainLoss {trainLoss:.3f}, evalLoss: {evalLoss:.3f}')

    elif modelname == "rfr":
        n_estimators = trial.suggest_int("n_estimators",10,1000)
        max_features = trial.suggest_categorical("max_features",["sqrt","log2",None])
        max_depth = trial.suggest_int("max_depth",10,100)
        model = rfr(n_estimators=n_estimators,max_features=max_features,max_depth=max_depth,verbose=1)
        with parallel_backend('threading', n_jobs=-1):
            model.fit(xTrT[idxSel], yTrT[idxSel].reshape(-1))
            trainLoss = mse(yTrT[idxSel].reshape(-1),model.predict(xTrT[idxSel]).reshape(-1))
            evalLoss = mse(yET.reshape(-1),model.predict(xET).reshape(-1))
            print(modelname,f' TrainLoss {trainLoss:.3f}, evalLoss: {evalLoss:.3f}')

    elif modelname == "gradB":
        learning_rate = trial.suggest_float("learning_rate",1e-4,1e0,log=True)
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes",1,500)
        max_depth = trial.suggest_int("max_depth",10,100)
        model = gradB(learning_rate=learning_rate,max_leaf_nodes=max_leaf_nodes,max_depth=max_depth,verbose=1)
        with parallel_backend('threading', n_jobs=-1):
            model.fit(xTrT[idxSel], yTrT[idxSel].reshape(-1))
            trainLoss = mse(yTrT[idxSel].reshape(-1),model.predict(xTrT[idxSel]).reshape(-1))
            evalLoss = mse(yET.reshape(-1),model.predict(xET).reshape(-1))
            print(modelname,f' TrainLoss {trainLoss:.3f}, evalLoss: {evalLoss:.3f}')
    return evalLoss


def objective(trial):

    model = makeModel(trial).to(device)
    lr = trial.suggest_float("lr", minLr, maxLr, log=True)

    opti = t.optim.Adam(model.parameters(),lr = lr)
    evalLossMean = trainNN(model,opti,trDl, evalDl,epochs=epochs, trial=trial)
    return evalLossMean

def trainNN(nnNet,opti,trDl, evalDl,epochs=2, trial=None):

    writerCount=0
    evalWriterCount=0
    trLosses =[]
    evalLosses = []
    if trial is not None:
        L1val = trial.suggest_float("L1val",l1min,l1max,log=True)
    for e in tqdm(range(epochs)):
        epochLossTr = []
        epochLossE = []
        for i, (x,y) in enumerate(trDl):
            nnNet.train()
            out = nnNet(x.to(device))
            lossTr = criterion(y.to(device), out)
            if addL1Reg:
                    l2_reg = t.tensor(0.).to(device)
                    for nParam, parameter in enumerate(nnNet.parameters()):
                        l2_reg += t.linalg.norm(parameter)
                    lossTr += L1val * l2_reg
            opti.zero_grad()
            lossTr.backward()
            opti.step()
            epochLossTr.append(lossTr.cpu().detach().numpy().item())
            if logTB:
                writerCount+=1
                writerTr.add_scalar("lossTr", lossTr.cpu().detach().numpy().item(),writerCount)
            if trial is not None and i >= numBatchesForOptunaTr*batchSize:
                break


            if i%500==0:
                with t.no_grad():
                    for iTe,(xTe, yTe) in enumerate(evalDl):
                        nnNet.eval()
                        outTe = nnNet(xTe.to(device))
                        lossTe = criterion(yTe.to(device),outTe)
                        if addL1Reg:
                            l2_reg = t.tensor(0.).to(device)
                            for parameter in nnNet.parameters():
                                l2_reg += t.linalg.norm(parameter)
                            lossTe += L1val * l2_reg
                        epochLossE.append(lossTe.cpu().detach().numpy().item())
                        if logTB:
                            evalWriterCount +=1
                            writerEval.add_scalar("lossEval", lossTe.cpu().detach().numpy().item(),evalWriterCount)
                        if trial is not None and iTe >= numBatchesForOptunaTe*batchSize:
                            break
        trLosses.append(np.mean(epochLossTr))
        evalLosses.append(np.mean(epochLossE))

        if trial is not None:
            trial.report(np.mean(epochLossE), e)
            if trial.should_prune():
                raise opt.exceptions.TrialPruned()
        print('\nEpoch ',e,' mean eval Loss ',np.mean(epochLossE))

    # plot losses

    plt.plot(trLosses, 'k.',label='Train')
    plt.plot(evalLosses, 'r.',label='Eval')
    plt.xlabel('Epochs')
    plt.ylabel('Eval')
    plt.savefig(f'logs/{dateTimeNow}_MeanEvalLoss_{np.mean(epochLossE):.3f}_Losses.png',dpi=200)
    # plt.show()
    plt.close()

    return np.mean(epochLossE)

#Datadir
mainDir = r"E:\KaggleData\Tabular Playground Series - Jan 2021"

#Hyperparameters and toggles
epochs = 50
lr = 6e-6
batchSize = 1024
addL1Reg = True
L1val = 0.004
doLearningCurve = False

logTB = 0

doShallows =False
doPYNN = True
doLightning = 0
tuneModel = 0

doOptuna = True
#optuna Params
possibleActiFuncs = ["ReLU"] #,"Sigmoid","LeakyReLU","Tanh"
minLayers = 1
maxLayers = 6
minNeurons = 20
maxNeurons = 500 #TODO try powers of 2 with int suggest
minDropOut = 0.
maxDropOut = 0.7
l1min = 1e-7
l1max = 1e-2
minLr = 1e-8
maxLr = 1e-3
maxTrials = 100
maxTime = 60000
numBatchesForOptunaTr = 30
numBatchesForOptunaTe = 20

#optuna shallow params
possibleShallows = ["rfr","gradB"]#,"SVR","bayR","gradB"
numShallowSamples = 50000

#Toggles for data overview
doPCA = False #looks to be unhelpful
plotPCA = False
checkHist = False #Target data looks like double normal distribution
plotCorreclations = False

os.chdir(mainDir)
dateTimeNow =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if logTB:
    writerTr = SummaryWriter("logs/"+dateTimeNow+"TrainLosses")
    writerEval = SummaryWriter("logs/"+dateTimeNow+"EvalLosses")

trainDataRaw = pd.read_csv("train.csv")
testDataRaw = pd.read_csv("test.csv")

targetCol = "target"
idCol = "id"

trD = trainDataRaw.drop([targetCol, idCol],axis=1)
teD = testDataRaw.drop([idCol],axis=1)
trTarget = trainDataRaw[targetCol]

if doPCA:
    pca = PCA(n_components=10).fit(trD)
    trD = pca.transform(trD)
    teD = pca.transform(teD)
    print(trD.shape,teD.shape)
    if plotPCA:
        pcaPlot = PCA(n_components=3).fit(trainDataRaw.drop([targetCol, idCol],axis=1), y=trTarget)
        xPCAplot = pcaPlot.transform(trainDataRaw.drop([targetCol, idCol],axis=1))
        yPCAplot = trTarget.copy()


        #%matplotlib qt

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        numSamples = 1000
        idxArr = np.arange(len(xPCAplot))
        np.random.shuffle(idxArr)
        idxSel = idxArr[:numSamples]
        ax.scatter(xPCAplot[idxSel,0], xPCAplot[idxSel,1], xPCAplot[idxSel,2], c = yPCAplot[idxSel])
        plt.show()
        plt.close()

        #%matplotlib inline


if plotCorreclations:
    corrs = trainDataRaw.drop("id",axis=1).corr()
    corrPlotThreshold = 1 # to check for weak correlations, cont11 and 12 correlate highly
    sb.heatmap(corrs[(corrs < corrPlotThreshold)&(corrs > -corrPlotThreshold)])
    plt.savefig('./AllCorrelations.png',dpi=300)
    plt.show()
    plt.close()
    plt.show()
    plt.close()
    plt.plot(corrs.keys().drop("target"),corrs["target"].drop("target"),'k.')
    plt.ylabel('Correlation with target')
    plt.xticks(rotation=45)
    plt.savefig('./CorrelationsWithTarget.png',dpi=300)
    plt.show()
    plt.close()

if checkHist:
    plt.hist(trTarget,bins=200)
    plt.xlabel('Target Variable')
    plt.ylabel('Frequency')
    plt.savefig('./TargetData_Histogram.png',dpi=300)
    plt.show()
    plt.close()

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
dSetTr = TensorDataset(t.Tensor(xTrT), t.Tensor(yTrT))
dSetE = TensorDataset(t.Tensor(xET), t.Tensor(yET))

dSetTr2 = dset(xTrT, yTrT)
dSetE2 = dset(xET,yET)
trDl = dl(dSetTr,batch_size=batchSize,shuffle=True)
evalDl = dl(dSetE,batch_size=batchSize,shuffle=True)

# Shallow tests

if doShallows:

    if doOptuna:
        studyShallow = opt.create_study(direction="minimize")
        studyShallow.optimize(shallowObjective, maxTrials)
        shallowTrial = studyShallow.best_trial
        print(shallowTrial)
        with open(f'./Shallow_BestTrialParams{dateTimeNow}.txt','w+') as f:
            for k, v in shallowTrial.params.items():
                f.write(f"'{k}':{v}\n")
                print(f"'{k}':{v},")
    else:
        ranForest = rfr(verbose=1) # mse: 0.130517
        bay = bayR(verbose=1)
        svr = SVR(verbose=1)

        bay.fit(xTrT, yTrT)
        bay = mse(bay.predict(xTrT),yTrT)
        print('Forest MSE: ',np.sqrt(bay))




# nn approach
device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
writerCount = 0
evalWriterCount = 0
if doPYNN:

    if doOptuna:
        criterion = nn.MSELoss()
        study = opt.create_study(direction="minimize")
        study.optimize(objective, n_trials=maxTrials, timeout=maxTime)
        trial = study.best_trial
        print('\nBest Study Parameters:')

        with open(f'./BestTrialParams{dateTimeNow}.txt','w+') as f:
            for k, v in trial.params.items():
                f.write(f"'{k}':{v}\n")
                print(f"'{k}':{v},")
    elif doLearningCurve:
        indsTr =np.arange(len(dSetTr))
        np.random.shuffle(indsTr)
        subTrDl = dl(Subset(dSetTr,indsTr[:50000]),batch_size=batchSize,shuffle=True)

        nnNet = net(xTrT, yTrT).to(device)
        opti = t.optim.Adam(nnNet.parameters(), lr = lr)
        criterion = nn.MSELoss()

        evalLossFromTrain = trainNN(subTrDl, evalDl, epochs=3)
        print('Final Loss val ', evalLossFromTrain)

    else:
        nnNet = net(xTrT, yTrT).to(device)
        opti = t.optim.Adam(nnNet.parameters(), lr = lr)
        criterion = nn.MSELoss()

        evalLossFromTrain = trainNN(trDl, evalDl)
        print('Final Loss val ', evalLossFromTrain)

        nnNet.eval()
        submissionY = nnNet(t.Tensor(teDT).to(device)).cpu().detach().numpy()
        dfSubmission = pd.DataFrame(data=submissionY,
                                    index=np.arange(0,len(submissionY)))
        dfSubmission.reset_index(inplace=True)
        dfSubmission.columns = ['id','target']
        # dfSubmission.rename({'index':'id'},inplace=True)
        dfSubmission.to_csv(f'logs/{dateTimeNow}_Submission.csv',index=False)

## nn with LightningModule
if doLightning:
    trDl = dl(TensorDataset(t.Tensor(xTrT).float(), t.Tensor(yTrT).float()),
              batch_size=batchSize,shuffle=True)

    trainer = pl.Trainer(gpus=1,max_epochs=epochs,stochastic_weight_avg=True)
    model1 = LitModel(t.Tensor(xTrT[:50]).float())
    if tuneModel:
        lr_finder = trainer.tuner.lr_find(model1,trDl)
        fig = lr_finder.plot(suggest=True)
        fig.show()
        new_lr = lr_finder.suggestion()
        # trainer.tune(model1,trDl)
    else:
        trainer.fit(model1, trDl)

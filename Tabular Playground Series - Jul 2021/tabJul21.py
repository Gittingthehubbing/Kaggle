# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 20:26:19 2021

@author: Quiet
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 19:38:46 2021

@author: Quiet

Tabular Playground Series - Jan 2021
"""
import matplotlib.pyplot as plt
import torch as t
from torch.utils.data.dataloader import DataLoader as dl
from torch.utils.data import TensorDataset, Dataset, Subset
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
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler as ss
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_log_error as mslr
from joblib import parallel_backend # to improve sklearn training speed
import os
import datetime
from tensorboardX import SummaryWriter
from sklearn.decomposition import PCA
import optuna as opt
from optuna.integration import PyTorchLightningPruningCallback
import pickle

from scipy import stats


#TODO implement optuna for shallows
#TODO Rebalance data (Might not be needed)
#TODO increase NN complexity


class LitModel(pl.LightningModule):

    def __init__(self, x, trial=None):
        super().__init__()
        if trial is not None:
            self.lr = trial.suggest_float("lr", minLr, maxLr)
            self.weight_decay =  trial.suggest_float("weight_decay", minWD, maxWD)
        else:
            self.lr = bestDict["lr"]
            self.weight_decay =  bestDict["weight_decay"]
        self.model = makeModel(trial, bestDict)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True,on_epoch=True)
        return loss

    def configure_optimizers(self):
       return t.optim.Adam(self.parameters(), 
                           lr=(self.lr or self.learning_rate))

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss =F.mse_loss(y_hat, y)
        self.log('val_loss', val_loss, prog_bar=True,on_epoch=True)
        return val_loss



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


def makeModel(trial = None, hpDict = None):
    # sets layers and number of neurons for each trial
    #builds dict with parameters that can be used to make best model
    layers=[]
    inFeat = xTrT.shape[1] #assumes 1D data for each sample
    lastOut = yTrT.shape[1]
    if trial is not None:
        nl = trial.suggest_int("nL",minLayers,maxLayers)
        
        for i in range(nl):
            outFeat = trial.suggest_int(f"n{i}",minNeurons,maxNeurons)
            layers.append(nn.Linear(inFeat, outFeat))
            actiLayer = trial.suggest_categorical(f"a{i}",possibleActiFuncs)
            layers.append(getattr(nn, actiLayer)())
            dropOutRatio = trial.suggest_float(f"dropoutL{i}",minDropOut,maxDropOut)
            layers.append(nn.Dropout(dropOutRatio))
            inFeat = outFeat
            
    elif hpDict is not None:
        nl = hpDict["nL"]
        
        for i in range(nl):
            outFeat = hpDict[f"n{i}"]
            layers.append(nn.Linear(inFeat, outFeat))
            actiLayer = hpDict[f"a{i}"]
            layers.append(getattr(nn, actiLayer)())
            dropOutRatio = hpDict[f"dropoutL{i}"]
            layers.append(nn.Dropout(dropOutRatio))
            inFeat = outFeat
            
    layers.append(nn.Linear(inFeat, lastOut))

    return nn.Sequential(*layers)



def runShallowOpt(model, modelname,idxSel):
    with parallel_backend('threading', n_jobs=-1):
        model.fit(xTrT, yTrT)
    yTrPredInvTrans = predictAndInvTransform(xTrT, model)
    yEPredInvTrans = predictAndInvTransform(xET, model)

    mseLoss = mse(yEPredInvTrans, yE)
    rmsleLossTrain = RMSLE(yTrPredInvTrans, yTr)
    rmsleLossEval = RMSLE(yEPredInvTrans, yE)

    #predSub = predictAndInvTransform(teDT,model)
    
    print(modelname,f' TrainLoss {rmsleLossTrain:.3f}, evalLoss: {rmsleLossEval:.3f}')
    return rmsleLossTrain, rmsleLossEval


        
def shallowObjective(trial):

    modelname = trial.suggest_categorical("modelname",possibleShallows)
    idxArr = np.arange(len(xTrT))
    np.random.shuffle(idxArr)
    if numShallowSamples >= len(xTrT):
        idxSel = idxArr
    else:
        idxSel = idxArr[:numShallowSamples]
    if modelname == "SVR":
        C = trial.suggest_float("C",1e0,1e4,log=True)
        kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
        if kernel == 'poly':
            polyDeg = trial.suggest_int("degree",2,6)
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
        multiRegModel = MultiOutputRegressor(model)
        trainLoss, evalLoss = runShallowOpt(multiRegModel, modelname,idxSel)
        
    elif modelname == "rfr":
        n_estimators = trial.suggest_int("n_estimators",10,1000)
        max_features = trial.suggest_categorical("max_features",["sqrt","log2",None])
        max_depth = trial.suggest_int("max_depth",10,100)
        model = rfr(n_estimators=n_estimators,max_features=max_features,max_depth=max_depth,verbose=1)
        multiRegModel = MultiOutputRegressor(model)
        trainLoss, evalLoss = runShallowOpt(multiRegModel, modelname,idxSel)
        
    elif modelname == "gradB":
        learning_rate = trial.suggest_float("learning_rate",1e-4,1e0,log=True)
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes",1,500)
        max_depth = trial.suggest_int("max_depth",10,100)
        model = gradB(learning_rate=learning_rate,max_leaf_nodes=max_leaf_nodes,max_depth=max_depth,verbose=1)
        multiRegModel = MultiOutputRegressor(model)
        trainLoss, evalLoss = runShallowOpt(multiRegModel, modelname,idxSel)
        
    elif modelname == 'xgboost':
        # lambdaVal = trial.suggest_float("lambdaVal",1e-5,1e0,log=True)
        # alpha = trial.suggest_float("alpha",1e-5,1e0,log=True)
        # learning_rate = trial.suggest_float("learning_rate",1e-5,1e0,log=True)
        # max_depth = trial.suggest_int("max_depth",1,500)
        # n_estimators = trial.suggest_int("n_estimators",10,10000)
        # min_child_weight = trial.suggest_int("min_child_weight",1,500)
        paramDict = {
            "lambda" : trial.suggest_float("lambdaVal",1e-5,1e0,log=True),
            "alpha" : trial.suggest_float("alpha",1e-5,1e0,log=True),
            "learning_rate" : trial.suggest_float("learning_rate",1e-5,1e0,log=True),
            "max_depth" : trial.suggest_int("max_depth",1,20),
            "n_estimators" : trial.suggest_int("n_estimators",10,10000),
            "min_child_weight" : trial.suggest_int("min_child_weight",1,500),
            'tree_method':'gpu_hist',
            'predictor': 'gpu_predictor'
            }
        
        model = xgb.XGBRegressor(**paramDict)
        multiRegModel = MultiOutputRegressor(model)
        trainLoss, evalLoss = runShallowOpt(multiRegModel, modelname,idxSel)
    return evalLoss


def litObjective(trial):
    
    model = LitModel(xTrT[:1000],trial=trial)
    trainer= pl.Trainer(
        limit_val_batches = 0.2,
        logger = tb_logger, max_epochs=epochs, gpus=1,
        callbacks=[PyTorchLightningPruningCallback(
            trial, monitor="val_loss")])
    hyperP = trial.params
    trainer.logger.log_hyperparams(hyperP)
    trainer.fit(model,train_dataloader=trDl,val_dataloaders=evalDl)
    return trainer.callback_metrics["val_loss"].item()
    
    
def objective(trial):

    model = makeModel(trial, hpDict=None).to(device)
    lr = trial.suggest_float("lr", minLr, maxLr, log=True)
    weight_decay = trial.suggest_float("weight_decay",1e-8,1e-2, log=True)

    opti = t.optim.Adam(model.parameters(),lr = lr,
                        weight_decay=weight_decay)
    evalLossMean = trainNN(model,opti,trDl, evalDl,epochs=epochs,
                           trial=trial)
    return evalLossMean

def trainNN(nnNet,opti,trDl, evalDl,epochs=2, trial=None):

    writerCount=0
    evalWriterCount=0
    trLosses =[]
    evalLosses = []
    rmsle_eval_Losses = []
    print('Trial check: trial is None is ', trial is None)
    if 'bestDict' in globals():
        L1val = bestDict["L1val"]
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
        with t.no_grad():
            yEPredictedNN = predictAndInvTransform(
                     t.Tensor(xET),nnNet.to(t.device('cpu')), deepflag=True
                )
            nnNet.to(device)
            rmsle_eval = RMSLE(yEPredictedNN, yE)
            rmsle_eval_Losses.append(rmsle_eval)
        if trial is not None:
            trial.report(rmsle_eval, e)
            if trial.should_prune():
                raise opt.exceptions.TrialPruned()
        print('\nEpoch ',e,' mean train Loss ',np.mean(epochLossTr))
        print('\nEpoch ',e,' mean eval Loss ',np.mean(epochLossE))

    # plot losses

    plt.plot(trLosses, 'k.',label='Train')
    plt.plot(evalLosses, 'r.',label='Eval')
    plt.xlabel('Epochs')
    plt.ylabel('Eval')
    plt.savefig(f'logs/{dateTimeNow}_MeanEvalLoss_{np.mean(epochLossE):.3f}_Losses.png',dpi=200)
    # plt.show()
    plt.close()

    return np.mean(rmsle_eval_Losses)


def saveBestTrial(study,name):
    
    trial = study.best_trial
    print('\nBest Study Parameters:')

    with open(f'./BestTrialParams{dateTimeNow}_{name}.txt','w+') as f:
        for k, v in trial.params.items():
            f.write(f"'{k}':{v}\n")
            print(f"'{k}':{v},")
    with open(f'./BestTrialParams{dateTimeNow}_{name}.pkl','wb') as f:
        pickle.dump(trial.params, f)



#Datadir
mainDir = r"E:\KaggleData\Tabular Playground Series - Jul 2021"

"""
uses mean column-wise root mean squared logarithmic error
"""

#Hyperparameters and toggles
splitRatio = 0.95 #for train eval split
epochs = 500
lr = 6e-6
batchSize = 1024
addL1Reg = True
L1val = 0.004
weight_decay = 1e-4
doLearningCurve = False

logTB = True
logTB_lightining = True

doShallows =1
doPYNN = 0
doLightning = 0
tuneModel = 0

doOptuna = 1 #applies to all methods above

#optuna Params
possibleActiFuncs = ["ReLU"] #,"Sigmoid","LeakyReLU","Tanh"
minLayers = 1
maxLayers = 10
minNeurons = 20
maxNeurons = 1000 #TODO try powers of 2 with int suggest
minDropOut = 0.
maxDropOut = 0.7
l1min = 1e-7 #not used for lightning
l1max = 1e-2
minWD = 1e-6
maxWD = 5e-3 #used for lightning
minLr = 1e-7
maxLr = 1e-2
maxTrials = 120
maxTime = 1*60*60
numBatchesForOptunaTr = 30
numBatchesForOptunaTe = 20

#optuna shallow params
possibleShallows = ["rfr"]#"rfr","SVR","gradB","xgboost"
numShallowSamples = 50000

#Toggles for data augmentation overview
doPCA = False #looks to be unhelpful
doBoxCox = 1 # works quite well but requires to be fit to Training data
logRedistr = 0 # might not be the best
duplicateUnderrepData = True #Adds copies of data points to compensate deviation from normal distr
dupliFac = 3 #how many times the data should be appended

#create extra features from previous data points
pastPoints = 10 # this number -1 is number of new features per column

plotPCA = False
checkHist = 1 #Target data looks like double normal distribution
plotCorreclations = 1
printDataStats = 0


#best (BestTrialParams20210703-170634) currently causes overfitting
bestDict ={
    'nL':6,
    'n0':435,
    'a0':'ReLU',
    'dropoutL0':0.41252116154797547,
    'n1':234,
    'a1':'ReLU',
    'dropoutL1':0.3705802089015725,
    'n2':364,
    'a2':'ReLU',
    'dropoutL2':0.17624519916649536,
    'n3':493,
    'a3':'ReLU',
    'dropoutL3':0.010860518403366382,
    'n4':252,
    'a4':'ReLU',
    'dropoutL4':0.21194478383187768,
    'n5':500,
    'a5':'ReLU',
    'dropoutL5':0.03323032803874664,
    'lr':0.0004610912753128815,
    'L1val':1.395796158400521e-07
    }
with open(f'{mainDir}/BestTrialParams20210710-180700.pkl','rb') as f:
            bestDict = pickle.load(f)

    
    
os.chdir(mainDir)
dateTimeNow =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if logTB and doPYNN:
    writerTr = SummaryWriter("logs/"+dateTimeNow+"TrainLosses")
    writerEval = SummaryWriter("logs/"+dateTimeNow+"EvalLosses")
if logTB_lightining and doLightning:
    tb_logger = pl.loggers.TensorBoardLogger('lightning_logs/')

trainDataRaw = pd.read_csv("train.csv")
testDataRaw = pd.read_csv("test.csv")

targetCols = ["target_carbon_monoxide","target_benzene","target_nitrogen_oxides"]
idCol = "date_time"
trainingCols = list(trainDataRaw.columns)
for col in targetCols.copy():
    trainingCols.remove(col)
trainingCols.remove(idCol)

trD = trainDataRaw.drop([*targetCols,idCol],axis=1).copy()
teD = testDataRaw.copy().drop([idCol],axis=1)
trTarget = trainDataRaw[targetCols].copy()


if logRedistr:
    trTargetBefore = np.round(trTarget.copy(),10)
    trTarget = np.round(np.log1p(trTarget),10)
    trTargetRecovered = np.round(np.exp(trTarget)-1,10)
    print('Recovery difference: ', np.sum((trTargetRecovered-trTargetBefore)))


if duplicateUnderrepData:
    for i in range(dupliFac):
        extraData = trTarget[trTarget["target_benzene"]>=0.2]
        extraData = extraData.append(extraData)
    trTarget = trTarget.append(extraData)


# plt.hist(trTarget[trTarget["target_benzene"]<20]["target_benzene"],200)


def addPastDataFeatures(df,pastPoints,cols):
    """
    Pass cols as list even if single
    pastPoints is how far into the past to go
    """
    for p in range(1,pastPoints):
        for col in cols:
            oldColName = col
            newColName = f"{col}-{p}"
            df.at[:p-1,newColName] = df.at[0,oldColName]
            for i in range(p,len(df)):
                df.at[i,newColName] = df.at[i-p,oldColName]
    return df

allCols = list(trD.keys())
trD = addPastDataFeatures(trD, pastPoints, allCols)
teD = addPastDataFeatures(teD, pastPoints, allCols)

trD.to_html('trD.html')
        

if doPCA:
    pca = PCA(n_components=10).fit(trD)
    trD = pca.transform(trD)
    teD = pca.transform(teD)
    print(trD.shape,teD.shape)
    if plotPCA:
        pcaPlot = PCA(n_components=3).fit(trainDataRaw.drop([*targetCols,idCol],axis=1), y=trTarget)
        xPCAplot = pcaPlot.transform(trainDataRaw.drop([*targetCols,idCol],axis=1))
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

#from sklearn.model_selection import TimeSeriesSplit

#timesplitter used for validation not main training
#timeSplitter = TimeSeriesSplit(n_splits=numOfSeqs,max_train_size=seqLength,test_size=2)

# xTrain = np.empty((numOfSeqs, 10,8))
# yTrain = np.empty((numOfSeqs, 10,3))
# for i,(trainIndex, testIndex) in enumerate(timeSplitter.split(X=trD,y=trTarget)):

#     print('x data shape',trD.to_numpy()[trainIndex].shape, trD.to_numpy()[testIndex].shape)
#     print('y data shape',trTarget.to_numpy()[trainIndex].shape, trTarget.to_numpy()[testIndex].shape)
#     xTrain= trD.to_numpy()[trainIndex]
#     yTrain= trTarget.to_numpy()[trainIndex]
#     if i>5:break


#Random split should be ok, because time windows were added as features

idx = np.arange(0,len(trD))
np.random.shuffle(idx)
trainIdx = idx[:int(len(trD)*splitRatio)]
testIdx = idx[int(len(trD)*splitRatio):]
xTr =np.asarray(trD)[trainIdx]
xE= np.asarray(trD)[testIdx]
yTr = np.asarray(trTarget)[trainIdx]
yE = np.asarray(trTarget)[testIdx]


xTrT = xTr.copy()
xET = xE.copy()
teDT = teD.copy()

yTrT = yTr.copy()
yET = yE.copy()

if doBoxCox:   
    yTrTBefore = yTrT.copy()
    powTrans = PowerTransformer(method='box-cox',standardize=False)
    powTrans.fit(yTrT)
    yTrT = powTrans.transform(yTrT)
    yTrTRecovered = powTrans.inverse_transform(yTrT.copy())
    print('Recovery difference: ', np.sum((yTrTRecovered-yTrTBefore)))

    yET = powTrans.transform(yET)
      


if checkHist:
    fig, axs = plt.subplots(len(trTarget.columns),1,dpi=300,squeeze=False)
    for iC, c in enumerate(trTarget.columns):
        skew = trTarget[c].skew()
        axs[iC,0].hist(trTarget[c],bins=200)
        axs[iC,0].set_xlabel(c)
        axs[iC,0].set_ylabel('Frequency')
        axs[iC,0].set_title(rf"Skew of {c} is {skew:.4f}")
    fig.savefig(f'./TargetData_Histogram.png',dpi=300)
    plt.close()




if plotCorreclations:
    corrs = trainDataRaw.drop(idCol,axis=1).corr()
    corrPlotThreshold = 1 # to check for weak correlations, cont11 and 12 correlate highly
    sb.heatmap(corrs[(corrs < corrPlotThreshold)&(corrs > -corrPlotThreshold)])
    plt.savefig('./AllCorrelations.png',dpi=300)
    plt.close()
    plt.plot(corrs.keys().drop(targetCols),corrs[targetCols].drop(targetCols),'k.')
    plt.ylabel('Correlation with target')
    plt.xticks(rotation=45)
    plt.savefig('./CorrelationsWithTarget.png',dpi=300)
    plt.close()




# Normalise
ssTr = ss().fit(xTrT)

xTrT = ssTr.transform(xTrT)
xET = ssTr.transform(xET)
teDT = ssTr.transform(teD)

ssTarget = ss().fit(yTrT)
yTrT = ssTarget.transform(yTrT)
yET = ssTarget.transform(yET)

#Check stats
if printDataStats:
    print('\n raw train data stats:')
    trainDataRaw.drop(idCol,axis=1).info(verbose=1)
    print(trainDataRaw.drop(idCol,axis=1).describe())
    tempTrainDf = pd.DataFrame(data=xTrT,columns=trD.columns,index=range(len(xTrT)))
    tempTrainDf.info(verbose=1)
    print(tempTrainDf.describe())

#make Dataloaders
dSetTr = TensorDataset(t.Tensor(xTrT), t.Tensor(yTrT))
dSetE = TensorDataset(t.Tensor(xET), t.Tensor(yET))

dSetTr2 = dset(xTrT, yTrT)
dSetE2 = dset(xET,yET)
trDl = dl(dSetTr,batch_size=batchSize,shuffle=False)
evalDl = dl(dSetE,batch_size=batchSize,shuffle=False)

# Shallow tests

def RMSLE(ypred, yreal): #uses natural log
    ypred_log = np.log(np.clip(ypred+1,1e-6,1e10))
    yreal_log=np.log(yreal+1)
    diffSqr = np.square(ypred_log - yreal_log)
    rmsleLoss = np.sqrt(np.mean(diffSqr))
    return rmsleLoss

def predictAndInvTransform(x,model, deepflag=False):
    if deepflag:
        yPred = model(x)        
    else:
        yPred = model.predict(x)    
    yPredInv = ssTarget.inverse_transform(yPred)
    if logRedistr:
        yPredInv = np.exp(yPredInv)-1
    if doBoxCox:
        yPredInv = powTrans.inverse_transform(yPredInv)
    if np.isnan(yPredInv).sum()>0:
        yPredInv = np.nan_to_num(yPredInv,copy=True)
    return yPredInv

def runShallow(model):
    with parallel_backend('threading', n_jobs=-1):
        model.fit(xTrT, yTrT)
    #keep out of with statement to avoid kernel crash
    
    yTrPredInvTrans = predictAndInvTransform(xTrT, model)
    yEPredInvTrans = predictAndInvTransform(xET, model)

    mseLoss = mse(yEPredInvTrans, yE)
    rmsleLossTrain = RMSLE(yTrPredInvTrans, yTr)
    rmsleLossEval = RMSLE(yEPredInvTrans, yE)

    predSub = predictAndInvTransform(teDT,model)

    return model, mseLoss,  predSub, rmsleLossEval, rmsleLossTrain

def saveSubmission(data,name):
    dfSubmission = pd.DataFrame(data=data,
                                index=np.arange(0,len(data),1))
    dfSubmission.reset_index(inplace=True,drop=False)
    dfSubmission.columns = [idCol,*targetCols]
    dfSubmission[idCol] = testDataRaw[idCol]
    # dfSubmission.rename({'index':'id'},inplace=True)
    dfSubmission.to_csv(f'logs/{dateTimeNow}_{name}_Submission.csv',index=False)


if doShallows:

    if doOptuna:
        studyShallow = opt.create_study(direction="minimize")
        studyShallow.optimize(shallowObjective, maxTrials)
        shallowTrial = studyShallow.best_trial
        print(shallowTrial)
        saveBestTrial(studyShallow,'Shallow')        
        opt.visualization.plot_param_importances(studyShallow)
    else:
        
        with open(r'E:\KaggleData\Tabular Playground Series - Jul 2021/BestTrialParams20210710-190103_Shallow.pkl','rb') as f:
            paramDict=pickle.load(f)
        paramDict ['tree_method']='gpu_hist'
        paramDict ['predictor']='gpu_predictor' 
        if "lambdaVal" in paramDict.keys():
            paramDict["lambda"] = paramDict["lambdaVal"]
            del paramDict["lambdaVal"]
        if "modelname" in paramDict.keys():
            del paramDict["modelname"]
        #BestTrialParams20210704-145220_Shallow.pkl
        #multi reg seems to crash kernel
        multiRegXG = MultiOutputRegressor(xgb.XGBRegressor(**paramDict))
        print('Fitting XGBoost now')
        multiRegXG, mseLoss, predSub, rmsleLoss, rmsleLossTrain = runShallow(multiRegXG)
        print('XGBRegressor MSE: ',mseLoss)
        print('XGBRegressor RMSLE: ', rmsleLoss, ' Train: ',rmsleLossTrain)
        saveSubmission(predSub, 'Shallow_XGBoost')
        
        bestForestDict = {
        'modelname':rfr,
        'n_estimators':978,
        'max_features':'log2',
        'max_depth':81
        } 
        del bestForestDict["modelname"]
        multiRegForest = MultiOutputRegressor(
            rfr(**bestForestDict))

        bestSVR = {
            'modelname': 'SVR',
             'C': 35.542238529147916,
              'kernel': 'poly',
               'degree': 2,
                'gamma': 'scale',
                'coef0': 0.1829941183576717}
        del bestSVR["modelname"]
        svrMulti = MultiOutputRegressor(SVR(**bestSVR))
        ranSVR, mseLoss, predSub, rmsleLoss, rmsleLossTrain = runShallow(svrMulti)
        print('SVR MSE: ',mseLoss)
        print('SVR RMSLE: ', rmsleLoss, ' Train: ',rmsleLossTrain)
        saveSubmission(predSub, 'Shallow_SVR')

        ranForest, mseLoss, predSub, rmsleLoss, rmsleLossTrain = runShallow(multiRegForest)
        print('Forest MSE: ',mseLoss)
        print('Forest RMSLE: ', rmsleLoss, ' Train: ',rmsleLossTrain)
        saveSubmission(predSub, 'Shallow_Forest')

        #TODO take average of predictions
        


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
        with open(f'./BestTrialParams{dateTimeNow}.pkl','wb') as f:
            pickle.dump(trial.params, f)
        opt.visualization.plot_param_importances(study)
    elif doLearningCurve:
        indsTr =np.arange(len(dSetTr))
        np.random.shuffle(indsTr)
        subTrDl = dl(Subset(dSetTr,indsTr[:50000]),batch_size=batchSize,shuffle=True)

        nnNet = makeModel(bestDict).to(device)
        opti = t.optim.Adam(nnNet.parameters(), lr = bestDict["lr"])
        criterion = nn.MSELoss()

        evalLossFromTrain = trainNN(nnNet,opti,subTrDl, evalDl, epochs=10)
        print('Final Loss val ', evalLossFromTrain)

    else:
        nnNet = makeModel(hpDict=bestDict).to(device)
        opti = t.optim.Adam(
            nnNet.parameters(), lr = bestDict["lr"],
            weight_decay=1e-3)
        criterion = nn.MSELoss()

        evalLossFromTrain = trainNN(nnNet,opti,trDl, evalDl,epochs = epochs)
        print('Final Loss val ', evalLossFromTrain)

        nnNet.eval()
        with t.no_grad():
            yEPredictedNN = predictAndInvTransform(
                     t.Tensor(xET),nnNet.to(t.device('cpu')), deepflag=True
                )
            nnFinalEvalLoss = RMSLE(yEPredictedNN, yE)

            yTrPredictedNN = predictAndInvTransform(
                     t.Tensor(xTrT),nnNet.to(t.device('cpu')), deepflag=True
                )
            nnFinalTrainLoss = RMSLE(yTrPredictedNN, yTr)

            yPredNN = predictAndInvTransform(
                t.Tensor(teDT),nnNet.to(t.device('cpu')), deepflag=True)
            saveSubmission(yPredNN, 'DeepNN')
            print('\nNN model final RMSLE Loss:')
            print(f"Train: {nnFinalTrainLoss}, Eval: {nnFinalEvalLoss}")
       

## nn with LightningModule
if doLightning:
    
    if doOptuna:
        
        bestDict = None
        study = opt.create_study(direction="minimize", pruner=opt.pruners.MedianPruner())
        study.optimize(litObjective,n_trials=maxTrials, timeout=maxTime)
        saveBestTrial(study, 'Lightning')
        #Visualize parameter importances.
        opt.visualization.plot_param_importances(study)
    else:
        with open('BestTrialParams20210704-125835_Lightning.pkl','rb') as f:
            bestDict = pickle.load(f)
         
        trainer = pl.Trainer(
            gpus=1,max_epochs=epochs,stochastic_weight_avg=True, logger=tb_logger)
        model1 = LitModel(t.Tensor(xTrT[:50]).float())
        if tuneModel:
            lr_finder = trainer.tuner.lr_find(model1,trDl,evalDl)
            fig = lr_finder.plot(suggest=True)
            fig.savefig('LR_finder.png')
            new_lr = lr_finder.suggestion()
            # trainer.tune(model1,trDl)
        else:
            trainer.fit(model1, trDl, evalDl)
            with t.no_grad():
                yEPredictedLit = predictAndInvTransform(
                     t.Tensor(xET),model1, deepflag=True
                )
                litFinalEvalLoss = RMSLE(yEPredictedLit, yE)

                yTrPredictedLit = predictAndInvTransform(
                     t.Tensor(xTrT),model1, deepflag=True
                )
                litFinalTrainLoss = RMSLE(yTrPredictedLit, yTr)

                print('\nLighting model final RMSLE Loss:')
                print(f"Train: {litFinalTrainLoss}, Eval: {litFinalEvalLoss}")
                yPredLit = predictAndInvTransform(
                    t.Tensor(teDT),model1, deepflag=True)
                saveSubmission(yPredLit, 'DeepLit')

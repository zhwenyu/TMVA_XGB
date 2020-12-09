import os, sys
import time
import getopt
import argparse
import ROOT as r
import varsList
import numpy as np
import uproot 
from sklearn import metrics
#import shap
import pickle
import pandas as pd
import root_pandas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from root_pandas import to_root
from ROOT import TMVA
from ROOT import RDataFrame
import xgboost as xgb
from xgboost import XGBClassifier
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

#parser = argparse.ArgumentParser(description='Multivariate analysis for charged Higgs search')
#parser.add_argument("-k", "--varListKey", default="NewVar", help="Input variable list")
#parser.add_argument("-l", "--label", default="", help="label for output file")
#args = parser.parse_args()

DEFAULT_INFNAME  = "180"
DEFAULT_TREESIG  = "TreeS"
DEFAULT_TREEBKG  = "TreeB"
DEFAULT_NITS   = "10"
DEFAULT_MDEPTH   = "3"#str(len(varList))
DEFAULT_VARLISTKEY = "NewVar"
DEFAULT_SIGMASS = "M-3000"
DEFAULT_INTERACTIVE = True
note = '_4j_year2017'  # EDIT

shortopts  = "f:n:d:s:l:t:o:i:vh?"
longopts   = ["inputfile=", "nTrees=", "maxDepth=", "sigMass=", "varListKey=", "inputtrees=", "outputfile=","interactive=", "verbose", "help", "usage"]
opts, args = getopt.getopt( sys.argv[1:], shortopts, longopts )
    
infname     = DEFAULT_INFNAME
treeNameSig = DEFAULT_TREESIG
treeNameBkg = DEFAULT_TREEBKG
nIts        = DEFAULT_NITS
mDepth      = DEFAULT_MDEPTH
varListKey  = DEFAULT_VARLISTKEY
sigMass     = DEFAULT_SIGMASS
interactive = DEFAULT_INTERACTIVE
verbose     = True    

for o, a in opts:
    if o in ("-?", "-h", "--help", "--usage"):
        usage()
        sys.exit(0)
    elif o in ("-d", "--maxDepth"):
        mDepth = a
    elif o in ("-l", "--varListKey"):
        varListKey = a
    elif o in ("-f", "--inputfile"):
        infname = a
    elif o in ("-n", "--nIts"):
        nIts = a
    elif o in ("-o", "--outDir"):
        outDir = a
    elif o in ("-s", "--sigMass"):
        sigMass = a
    elif o in ("-i" , "--interactive"):
        interactive = a



def getFscore(model):
    fig,ax = plt.subplots()
    bst.get_score(importance_type='gain')
    xgb.plot_importance(bst, importance_type='gain', ax=ax,  max_num_features = 10, xlabel = 'Gain', ylabel = 'Var', title = 'XGB Training Output')
    plt.savefig('FScore_'+str(num_round)+'iterations'+"_"+str(param['max_depth'])+'depth_'+str(varListKey)+note+'.png')

def buildROC(target_test,test_preds):
    print target_test
    fpr, tpr, threshold = metrics.roc_curve(target_test, test_preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.yscale('log')
    plt.grid(1)
    plt.xlabel('False Positive Rate')
    plt.gcf().savefig('ROC_'+str(num_round)+'iterations'+"_"+str(param['max_depth'])+'depth_'+str(varListKey)+note+'.png')

selList = [["isTraining", ""], ["isElectron", ""], ["isMuon", ""],["DataPastTriggerX",""],["MCPastTriggerX"], ["minDR_lepJet", ""], ["leptonPt_MultiLepCalc"], ["NJets_JetSubCalc"], ["NJetsCSVwithSF_MultiLepCalc"], ["corr_met_MultiLepCalc"], ["MT_lepMet"], ["AK4HT"]]
weightList = [["pileupWeight", ""], ["lepIdSF", ""], ["EGammaGsfSF", ""], ["MCWeight_MultiLepCalc", ""], ["triggerXSF", ""], ["isoSF", ""], ["L1NonPrefiringProb_CommonCalc", ""], ["tthfWeight", ""], ["btagCSVWeight"], ["btagCSVRenormWeight"]] 
varList = varsList.varList[varListKey]
inputList = list(set(iVar[0] for iVar in varList+selList+weightList))

inputDir = varsList.inputDir
#infname = "TTTT_TuneCP5_13TeV-amcatnlo-pythia8_hadd.root" # 2018 # EDIT
infname = "TTTT_TuneCP5_PSweights_13TeV-amcatnlo-pythia8_hadd.root" # 2017
print "Loading Signal Sample"
sig_tree = uproot.open(inputDir+infname)["ljmet"]
sig_df = sig_tree.pandas.df(branches= inputList)

#Event Selection
print(sig_df[sig_df.index.duplicated()]) 
sig_selected = (sig_df["isTraining"]<3)&(sig_df["NJets_JetSubCalc"]>=4)&(sig_df["NJetsCSVwithSF_MultiLepCalc"]>=2)&( ((sig_df["leptonPt_MultiLepCalc"]>20)&(sig_df["isElectron"]==True))|((sig_df["leptonPt_MultiLepCalc"]>20)&(sig_df["isMuon"]==True)))&(sig_df["MCPastTriggerX"]==1)&(sig_df["DataPastTriggerX"]==1)&(sig_df["corr_met_MultiLepCalc"]>60)&(sig_df["MT_lepMet"]>60)&(sig_df["minDR_lepJet"] > 0.4)&(sig_df["AK4HT"] > 500)
sig_df = sig_df[sig_selected]

print "Loading Background Samples"
back_dfs = []

bkgList = varsList.bkg
print bkgList
for ibkg in bkgList:
    print ibkg
    bkg_tree = uproot.open(inputDir+ibkg)["ljmet"]
    bkg_df = bkg_tree.pandas.df(branches= inputList)
    print bkg_df
    bkg_selected = (bkg_df["isTraining"]<3)&(bkg_df["NJets_JetSubCalc"]>=4)&(bkg_df["NJetsCSVwithSF_MultiLepCalc"]>=2)&( ((bkg_df["leptonPt_MultiLepCalc"]>20)&(bkg_df["isElectron"]==True))|((bkg_df["leptonPt_MultiLepCalc"]>20)&(bkg_df["isMuon"]==True)))&(bkg_df["MCPastTriggerX"]==1)&(bkg_df["DataPastTriggerX"]==1)&(bkg_df["corr_met_MultiLepCalc"]>60)&(bkg_df["MT_lepMet"]>60)&(bkg_df["minDR_lepJet"] > 0.4)&(bkg_df["AK4HT"] > 500)
    bkg_df = bkg_df[bkg_selected]
    print bkg_df
    back_dfs.append(bkg_df)

#print back_dfs
bkgall_df = pd.concat(back_dfs)
del back_dfs

#compute Weights 
weightSig = sig_df['pileupWeight']*sig_df['lepIdSF']*sig_df['EGammaGsfSF']*sig_df['MCWeight_MultiLepCalc']/(abs(sig_df['MCWeight_MultiLepCalc']))*sig_df["triggerXSF"]*sig_df["isoSF"]*sig_df["L1NonPrefiringProb_CommonCalc"]*sig_df["tthfWeight"]*sig_df["btagCSVWeight"]*sig_df["btagCSVRenormWeight"]
weightBkg = bkgall_df['pileupWeight']*bkgall_df['lepIdSF']*bkgall_df['EGammaGsfSF']*bkgall_df['MCWeight_MultiLepCalc']/(abs(bkgall_df['MCWeight_MultiLepCalc']))*bkgall_df["triggerXSF"]*bkgall_df["isoSF"]*bkgall_df["L1NonPrefiringProb_CommonCalc"]*bkgall_df["tthfWeight"]*bkgall_df["btagCSVWeight"]*bkgall_df["btagCSVRenormWeight"]

sigtotalWeight = np.sum(weightSig)
bkgtotalWeight = np.sum(weightBkg)

#compute the weight ratio to balance the training for XGBoost
scale = float(bkgtotalWeight)/float(sigtotalWeight)

randnum = np.random.rand(sig_df.shape[0]+bkgall_df.shape[0])
isTrain = randnum>0.2

#assign label
sig_df.loc[:,"isSignal"] = np.ones(sig_df.shape[0])
bkgall_df.loc[:, "isSignal"] = np.zeros(bkgall_df.shape[0])

dfall = pd.concat([sig_df, bkgall_df])

weightall = np.append(weightSig, weightBkg)

print dfall

train_var = []
for ivar in varList:
    train_var.append(ivar[0])
train_var.append("isSignal")
dfall_var = dfall.loc[:, train_var]

NDIM = len(dfall_var.columns)
dataset = dfall_var.values

X = dataset[:, 0:NDIM-1]
Y = dataset[:, NDIM-1]

dfall_var.loc[:, "isTrain"] = isTrain
dfall_var.loc[:, "Weight"] = weightall
print Y


#X_train_val, X_test, Y_train_val, Y_test, weight_train_val, weight_test = train_test_split(X, Y, weightall, test_size=0.3, random_state=7)

X_train_val = X[isTrain]
Y_train_val = Y[isTrain]
weight_train_val = weightall[isTrain]

X_test = X[isTrain==False]
Y_test = Y[isTrain==False]
weight_test = weightall[isTrain==False]

features = train_var[0:NDIM-1]
print features
print len(features)
print X_test.shape[1]
#del isTrain
#del weightall
#del dfall

dall = xgb.DMatrix(X, feature_names=features)
dtrain = xgb.DMatrix(X_train_val, label=Y_train_val, weight=weight_train_val, feature_names=features)
dtest = xgb.DMatrix(X_test, label=Y_test, weight=weight_test, feature_names=features)
watchlist = [(dtrain, 'train'), (dtest, 'eval')]#[(dtest, 'eval'), (dtrain, 'train')]
param = {
    'max_depth': int(mDepth),  # the maximum depth of each tree
    'eta': 0.1,  # the training step for each iteration
    'silent': 0,  # logging mode - quiet
    'TREE_METHOD': 'gpu_hist', ## Comment this out to run on non-gpu nodes
    'objective': 'binary:logistic',  # error evaluation for classification training
    'scale_pos_weight': scale,
    'eval_metric': 'auc',
    'subsample': 0.8
    }  # the number of classes that exist in this datset
num_round = int(nIts)  # the number of training iterations

bst = xgb.train(param, dtrain, num_round, watchlist, callbacks=[xgb.callback.print_evaluation()], early_stopping_rounds=10)
#bst = xgb.train(param, dtrain, num_round, nfold=5, metrics={'auc'}, seed=10)
dfall_var.loc[:, 'XGB'] = bst.predict(dall)
pickle.dump(bst,open(str(num_round)+'iterations'+"_"+str(param['max_depth'])+'depths_signal_region_'+str(varListKey)+note+'.dat','wb'))
bst.dump_model('dump.raw'+str(num_round)+'iterations'+"_"+str(param['max_depth'])+'depths_signal_region_'+str(varListKey)+note+'.txt')
bst.save_model('XGB'+str(num_round)+'iterations'+"_"+str(param['max_depth'])+'depths_signal_region_'+str(varListKey)+note+'.model')
#print dfall_var

dfall_var.to_root("XGB_classification_"+str(num_round)+'iterations'+"_"+str(param['max_depth'])+'depths_'+str(varListKey)+note+".root", key="XGB_Tree")


if interactive:
    predictions = bst.predict(dtest)
    buildROC(Y_test,predictions)
    getFscore(bst)
else:
    predictions = bst.predict(dtest)
    buildROC(Y_test,predictions)
    
### Plot Variable Importance Information
# fig,ax = plt.subplots(figsize=(4,5))
# bst.get_score(importance_type='gain')
# xgb.plot_importance(bst, importance_type='gain', ax=ax,  max_num_features = 10, xlabel = 'Gain', ylabel = 'Var', title = 'XGB Training Output')
# ax.grid(b = False)
# plt.tight_layout()
# plt.savefig('/plots/XGB'+str(num_round)+'iterations'+"_"+str(param['max_depth'])+'depths_signal_region_'+str(varListKey)+'_'+sigMass+'.png')

# t0 = time.time()
# print "Begining SHAP explainer"
# explainer = shap.TreeExplainer(bst)
# shap_values = explainer.shap_values(X)
# t1 = time.time()
# print t1-t0 + "Seconds?"
# shap.summary_plot(shap_values, "placeholder", plot_type = "bar")


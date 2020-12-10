import os, sys
import time
import getopt
import argparse
import ROOT
import array
import varsList
import numpy as np
import uproot
import pandas as pd
import math 
from math import sqrt
#import root_pandas
#from root_pandas import to_root
#from ROOT import TMVA
#from ROOT import RDataFrame
import xgboost as xgb
from xgboost import XGBClassifier
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

############################################################
# Copy the entire step-2 TTree, add XGB, save to a new TTree
############################################################

parser = argparse.ArgumentParser(description='Apply XGB for charged Higgs search')
parser.add_argument("-l", "--varListKey", default="NewVar", help="Input variable list")
parser.add_argument("-f", "--file", default="ChargedHiggs_HplusTB_HplusToTB_M-1000_13TeV_amcatnlo_pythia8_hadd.root", help="The name of the input file")
parser.add_argument("-m", "--model", default=""),
parser.add_argument("-o", "--output", default="NewOutput", help="The label for the output file")

args = parser.parse_args()

varListKey = args.varListKey
varList = varsList.varList[varListKey]
#inputDir = varsList.inputDir
infname = args.file
model = args.model

def Reshape(x):
   y = 0.5*(sqrt( (x+1)/2 ) + math.pow((x+1)/2, 12))
   return y 

print("Load Input File")

bst = xgb.Booster()
bst.load_model(model)

#bst300_3b6j = xgb.Booster()
#bst300_3b6j.load_model("XGB400iterations_2depths_0.05signal_region_NewVar_Re_M3003b6j.model")
#
#bst500 = xgb.Booster()
#bst500.load_model("XGB700iterations_3depths_0.05signal_region_NewVar_test_M500.model")
#
#bst500_3b6j = xgb.Booster()
#bst500_3b6j.load_model("XGB700iterations_2depths_0.05signal_region_NewVar_Re_M5003b6j.model")
#
#bst800 = xgb.Booster()
#bst800.load_model("XGB700iterations_3depths_0.05signal_region_NewVar_test_M800.model")
#
#bst800_3b6j = xgb.Booster()
#bst800_3b6j.load_model("XGB700iterations_2depths_0.05signal_region_NewVar_Re_M8003b6j.model")
#
#bst1000 = xgb.Booster()
#bst1000.load_model("XGB700iterations_3depths_0.05signal_region_NewVar_test_M1000.model")
#
#bst1000_3b6j = xgb.Booster()
#bst1000_3b6j.load_model("XGB700iterations_2depths_0.05signal_region_NewVar_Re_M10003b6j.model")
#
#bst1500 = xgb.Booster()
#bst1500.load_model("XGB700iterations_3depths_0.05signal_region_NewVar_test_M1500.model")
#
#bst1500_3b6j = xgb.Booster()
#bst1500_3b6j.load_model("XGB700iterations_2depths_0.05signal_region_NewVar_Re_M15003b6j.model")

train_var = []
for ivar in varList:
    train_var.append(ivar[0])
sig_tree = uproot.open(infname)["ljmet"]
numentries = sig_tree.numentries

tfile = ROOT.TFile.Open(infname)
ttree = tfile.Get("ljmet")

outputname = args.output
newfile = ROOT.TFile(outputname, "RECREATE")
newfile.cd()

newtree = ttree.CloneTree(0) 

XGB = array.array('d', [0])
#XGB300_3b6j = array.array('d', [0])
#XGB500 = array.array('d', [0])
#XGB500_3b6j = array.array('d', [0])
#XGB800 = array.array('d', [0])
#XGB800_3b6j = array.array('d', [0])
#XGB1000 = array.array('d', [0])
#XGB1000_3b6j = array.array('d', [0])
#XGB1500 = array.array('d', [0])
#XGB1500_3b6j = array.array('d', [0])



XGB_RS = array.array('d', [0])
#XGB300_3b6j_RS = array.array('d', [0])
#XGB500_RS = array.array('d', [0])
#XGB500_3b6j_RS = array.array('d', [0])
#XGB800_RS = array.array('d', [0])
#XGB800_3b6j_RS = array.array('d', [0])
#XGB1000_RS = array.array('d', [0])
#XGB1000_3b6j_RS = array.array('d', [0])
#XGB1500_RS = array.array('d', [0])
#XGB1500_3b6j_RS = array.array('d', [0])

newtree.Branch("XGB", XGB, "XGB/D")
#newtree.Branch("XGB300_3b6j", XGB300_3b6j, "XGB300_3b6j/D")
#newtree.Branch("XGB500", XGB500, "XGB500/D")
#newtree.Branch("XGB500_3b6j", XGB500_3b6j, "XGB500_3b6j/D")
#newtree.Branch("XGB800", XGB800, "XGB800/D")
#newtree.Branch("XGB800_3b6j", XGB800_3b6j, "XGB800_3b6j/D")
#newtree.Branch("XGB1000", XGB1000, "XGB1000/D")
#newtree.Branch("XGB1000_3b6j", XGB1000_3b6j, "XGB1000_3b6j/D")
#newtree.Branch("XGB1500", XGB1500, "XGB1500/D")
#newtree.Branch("XGB1500_3b6j", XGB1500_3b6j, "XGB1500_3b6j/D")

newtree.Branch("XGB_RS", XGB_RS, "XGB_RS/D")
#newtree.Branch("XGB300_3b6j_RS", XGB300_3b6j_RS, "XGB300_3b6j_RS/D")
#newtree.Branch("XGB500_RS", XGB500_RS, "XGB500_RS/D")
#newtree.Branch("XGB500_3b6j_RS", XGB500_3b6j_RS, "XGB500_3b6j_RS/D")
#newtree.Branch("XGB800_RS", XGB800_RS, "XGB800_RS/D")
#newtree.Branch("XGB800_3b6j_RS", XGB800_3b6j_RS, "XGB800_3b6j_RS/D")
#newtree.Branch("XGB1000_RS", XGB1000_RS, "XGB1000_RS/D")
#newtree.Branch("XGB1000_3b6j_RS", XGB1000_3b6j_RS, "XGB1000_3b6j_RS/D")
#newtree.Branch("XGB1500_RS", XGB1500_RS, "XGB1500_RS/D")
#newtree.Branch("XGB1500_3b6j_RS", XGB1500_3b6j_RS, "XGB1500_3b6j_RS/D")



iev=0
#XGB300_arrays = []
#XGB500_arrays = []
#XGB800_arrays = []
#XGB1000_arrays = []

print "Compute XGB"

for chunk in sig_tree.iterate("*", entrysteps=10000, namedecode="utf-8"):
    array_var=[]
    for var in train_var:
        array_var.append(chunk[var])
    dataset = np.column_stack(array_var)
    dX = xgb.DMatrix(dataset, feature_names=train_var)
    XGBpred = bst.predict(dX)
#    XGB300_3b6jpred = bst300_3b6j.predict(dX)
#    XGB500pred = bst500.predict(dX)
#    XGB500_3b6jpred = bst500_3b6j.predict(dX)
#    XGB800pred = bst800.predict(dX)
#    XGB800_3b6jpred = bst800_3b6j.predict(dX)
#    XGB1000pred = bst1000.predict(dX)
#    XGB1000_3b6jpred = bst1000_3b6j.predict(dX)
#    XGB1500pred = bst1500.predict(dX)
#    XGB1500_3b6jpred = bst1500_3b6j.predict(dX)
    #ichunk+=1
    chunck_size = len(XGBpred)
    for k in range(chunck_size):
        if iev%100000==0:
            print(iev)
        XGB[0] = XGBpred[k]
        XGB_RS[0] = Reshape(XGB[0]) 
#        XGB300_3b6j[0] = XGB300_3b6jpred[k]
#        XGB300_3b6j_RS[0] = Reshape(XGB300_3b6j[0])
# 
#        XGB500[0] = XGB500pred[k]
#        XGB500_RS[0] = Reshape(XGB500[0]) 
#        XGB500_3b6j[0] = XGB500_3b6jpred[k]
#        XGB500_3b6j_RS[0] = Reshape(XGB500_3b6j[0])
#
#        XGB800[0] = XGB800pred[k]
#        XGB800_RS[0] = Reshape(XGB800[0]) 
#        XGB800_3b6j[0] = XGB800_3b6jpred[k]
#        XGB800_3b6j_RS[0] = Reshape(XGB800_3b6j[0])
#
#        XGB1000[0] = XGB1000pred[k]
#        XGB1000_RS[0] = Reshape(XGB1000[0]) 
#        XGB1000_3b6j[0] = XGB1000_3b6jpred[k]
#        XGB1000_3b6j_RS[0] = Reshape(XGB1000_3b6j[0])
#
#        XGB1500[0] = XGB1500pred[k]
#        XGB1500_RS[0] = Reshape(XGB1500[0]) 
#        XGB1500_3b6j[0] = XGB1500_3b6jpred[k]
#        XGB1500_3b6j_RS[0] = Reshape(XGB1500_3b6j[0])


        ttree.GetEntry(iev)
        newtree.Fill()
        iev+=1

#nentries = ttree.GetEntries()
#
#    idx = iev%10000
#    XGB800[0] = XGB800_arrays[nchunk][idx]
#    XGB1000[0] = XGB1000_arrays[nchunk][idx]
#    newtree.Fill()

print newtree.GetEntries()
newfile.WriteTObject(newtree, "ljmet")
newfile.Close()


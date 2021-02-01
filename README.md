# TMVA_XGB
XGBOOST training and application for fourtops analysis.


Training: 

varsList.py -- check inputDir, varsList dictionary, file list

UpROOTXGB.py
Edit: note, selList, weightList, infname, sig_selected, bkg_selected, weightSig, weightBkg 
(other parameters possible) 
```
python UpROOTXGB.py -l ${varListKey} -n ${numInterations}
```
  

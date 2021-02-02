# TMVA_XGB
XGBOOST training and application for fourtops analysis.


Training: 

varsList.py -- check inputDir, varsList dictionary, file list

UpROOTXGB.py   
Edit: note, selList, weightList, infname, sig_selected, bkg_selected, weightSig, weightBkg 
(other parameters possible)  
One needs to test and find the best interation number.
```
python UpROOTXGB.py -l ${varListKey} -n ${numInterations}
```
 
Application:

varsList.py -- check varsList dictionary

submitApplication.py  
Edit: modelFile, inputDir, outputDir, condorDir
```
./steerCondorApplication.sh
```

import os,shutil,datetime,time,sys
import getpass
import varsList
from ROOT import *
from XRootD import client

###############################################

shift = sys.argv[1]
runDir = os.getcwd()
start_time = time.time()

modelFile = 'XGB500iterations_3depths_signal_region_SepRank6j73vars2017year40top_6j_year2018_NJetsCSV'
inputDir='/mnt/hadoop/store/group/bruxljm/FWLJMET102X_1lep2018_Oct2019_4t_01212021_step2/'+ shift + '/'
outputDir= '/mnt/hadoop/store/group/bruxljm/FWLJMET102X_1lep2018_Oct2019_4t_01272021_step3_wenyu/'+ modelFile +'/'+ shift + '/' # or 2018
condorDir= runDir+'/condor_logs/FWLJMET102X_1lep2018_Oct2019_4t_01272021_step3_wenyu/' + modelFile +'/'+ shift + '/' 
vListKey = modelFile.split('_')[4]

print 'Starting submission'
count=0

#inDir=inputDir[10:]
#outDir=outputDir[10:]

rootfiles = os.popen('ls '+inputDir)
os.system('mkdir -p '+outputDir)
os.system('mkdir -p '+condorDir)
#eosindir = inputDir[inputDir.find("/store"):]
#eosindir = "root://cmseos.fnal.gov/"+eosindir
#
#eosoutdir = outputDir[outputDir.find("/store"):]
#eosoutdir = "root://cmseos.fnal.gov/"+eosoutdir

for file in rootfiles:
    if 'root' not in file: continue
    #if not 'TTToSemiLeptonic' in file: continue
    #if not 'ttjj' in file: continue
#    if 'TTTo' in file: continue
    rawname = file[:-6]
    count+=1
    dict={'RUNDIR':runDir, 'CONDORDIR':condorDir, 'INPUTDIR':inputDir, 'FILENAME':rawname, 'OUTPUTDIR':outputDir, 'VLIST':vListKey, 'MODEL':modelFile}
    jdfName=condorDir+'/%(FILENAME)s.job'%dict
    print jdfName
    jdf=open(jdfName,'w')
    jdf.write(
"""universe = vanilla
Executable = %(RUNDIR)s/submitApplication.sh
Request_memory = 3000
Should_Transfer_Files = YES
WhenToTransferOutput = ON_EXIT
Transfer_Input_Files = %(RUNDIR)s/XGBApply.py, %(RUNDIR)s/varsList.py 
Output = %(FILENAME)s.out
Error = %(FILENAME)s.err
Log = %(FILENAME)s.log
JobBatchName = xgb_step3
Notification = Never
Arguments =  %(INPUTDIR)s/%(FILENAME)s.root  %(OUTPUTDIR)s  %(FILENAME)s.root  %(VLIST)s %(MODEL)s.model 
Queue 1"""%dict)
    jdf.close()
    os.chdir('%s/'%(condorDir))
    os.system('condor_submit %(FILENAME)s.job'%dict)
    os.system('sleep 0.5')
    os.chdir('%s'%(runDir))
    print count, "jobs submitted!!!"

print("--- %s minutes ---" % (round(time.time() - start_time, 2)/60))



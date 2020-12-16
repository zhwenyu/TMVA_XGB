import ROOT
from sklearn import metrics
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    plt.gcf().savefig('ROC_'+str(num_round)+'iterations'+"_"+str(param['max_depth'])+'depth_'+str(varListKey)+'_'+sigMass+'.png')

def build2ROC(target_train, pred_train, target_test, pred_test, name):
    fpr_1, tpr_1, _ = metrics.roc_curve(target_train, pred_train)
    roc_auc_1 = metrics.auc(fpr_1, tpr_1)
    fpr, tpr, threshold = metrics.roc_curve(target_test, pred_test)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr_1, tpr_1, 'orange', label = 'train AUC = %0.3f' % roc_auc_1)
    plt.plot(fpr, tpr, 'b', label = 'eval AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.yscale('log')
    plt.grid(1)
    plt.xlabel('False Positive Rate')
    plt.gcf().savefig('ROC_'+ name +'.png')
    plt.clf()

inputDir = 'output/'

for year in ['2017', '2018']:
  for nj in ['4', '6']:
    for nvars in ['40top', '50top']:

      inF = 'XGB_classification_500iterations_3depths_SepRank6j73vars2017year' + nvars + '_' + nj + 'j_year' + year
      print inF
      ttree = uproot.open(inputDir + inF + '.root')["XGB_Tree"]
      
      y = ttree.pandas.df(branches = ["isTrain", "isSignal", "XGB"])
      print y.shape
      y_train = y[y["isTrain"] == 1]["isSignal"]
      y_train_pred = y[y["isTrain"] == 1]["XGB"]
      y_eval = y[y["isTrain"] == 0]["isSignal"]
      y_eval_pred = y[y["isTrain"] == 0]["XGB"]
      build2ROC(y_train, y_train_pred, y_eval, y_eval_pred, inF.split('classification_')[1])

#%% Importing Packages
import pandas as pd
import numpy as np
import time
from pickle import load
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import copy
import scipy
#from sklearn.metrics import plot_confusion_matrix
import seaborn as sn
#import tensorflow
import winsound

#import git
import json
import os

#Import Sklearn Packages
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Import Sktime
#from sktime.classification.compose import TimeSeriesForestClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier, ElasticEnsemble, ProximityForest
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.hybrid import HIVECOTEV1
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.sklearn._rotation_forest import RotationForest
from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor
from sktime.utils.slope_and_trend import _slope

#Import own Packages
from Classification.custom_classifiers.classifiers import custom_fbeta 
from Classification.custom_classifiers.utils import build_sktime_data, calc_accuracy, data_stats
from Classification.data_handling.basics import read_in_data, handling_data, map_to_plaintext_labels
#from Classification.custom_classifiers.train_model import rise_training

#%% General

# Technical: Define max. time series length
series_length = 1000 #Time series length for predictions

#%% 1. Load OS Data
##############################################################################

#%% 1.1GVL and Load OS Data

#First load spydata df into workspace
#Required manual import
try:
    sensor_dic
except NameError:
    print("Please import OS Data")


#%% Multi-class predictions - base
#-----------------------------------------------------------

#%% Implement Multiclass prediction for OS-Data

#build Dataframe in correct format for sktime. Sensor length set to 1000 in build_sktime_data
keys_sorted, all_data = build_sktime_data(sensor_dic, series_length)
X, y = all_data['X'], all_data['y']

#Test train split with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, stratify=y, random_state=123)

#Convert to DataFrame
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

#Print overall and test/train stats
overall_stats = data_stats(keys_sorted, all_data['y'])
tt_data_stats = data_stats(keys_sorted, y_train, y_test)

print(overall_stats)

#%% Multiclass HIVECOTE 2 - base


clf = HIVECOTEV2(
    stc_params={
        "estimator": RotationForest(n_estimators=3),
        "n_shapelet_samples": 500, 
        "max_shapelets": 20,
        "batch_size": 100,
    },
    drcif_params={"n_estimators": 10},
    arsenal_params={"num_kernels": 100, "n_estimators": 5},
    tde_params={
        "n_parameter_samples": 25,
        "max_ensemble_size": 5,
        "randomly_selected_params": 10,
    },
    random_state=123
)

#Train model
clf.fit(X_train, y_train)

#Predict
y_pred = clf.predict(X_test)
cm_hivecote = confusion_matrix(y_test, y_pred)

#Analyse
#calc precision of baseline model
true_pos = np.diag(cm_hivecote)
false_pos = np.sum(cm_hivecote, axis=0) - true_pos

#Exclude zero divisors from precision calc
divisor_precision = true_pos + false_pos

#Calc average class precision
precision = np.average(true_pos[divisor_precision != 0]/divisor_precision[divisor_precision != 0])
acc = accuracy_score(y_test, y_pred)

#Baseline: Achieves precision of 63% and Accuracy of 59%


#%% Multiclass HIVECOTE 2 - base - Top 3
#Train Model
beta = 0.5
clf = HIVECOTEV2(
    stc_params={
        "estimator": RotationForest(n_estimators=3),
        "n_shapelet_samples": 500, 
        "max_shapelets": 20,
        "batch_size": 100,
    },
    drcif_params={"n_estimators": 10},
    arsenal_params={"num_kernels": 100, "n_estimators": 5},
    tde_params={
        "n_parameter_samples": 25,
        "max_ensemble_size": 5,
        "randomly_selected_params": 10,
    },
    random_state=123
)

#Train model
clf.fit(X_train, y_train)

#Predict with top three predictions: Based on MPMs work
yhat = clf.predict(X_test)

#get top3 prediction
yhat_proba = clf.predict_proba(X_test)
yhat_top3_type = np.argsort(yhat_proba, axis=1)[:,::-1][:,:3]
    
#for cases where algorithm rejects device: prob value -> 0, type -> 9
rej = (yhat != 9)
yhat_top3_type[np.invert(rej)] = 9

#compare test labels with top3 predictions
compare_matrix = np.tile(np.array(y_test)[:,None],(1,3)) == yhat_top3_type
correct_prediction_mask = np.amax(compare_matrix, axis=1, keepdims = False)

#put together a matrix to feed it into the usual scorer
#take all in the Top3 included labels from y_test, for all other entries just take the wrong predicted class or rejection
y_pred_top3 = (y_test*correct_prediction_mask) + (yhat*np.invert(correct_prediction_mask))
results_fbeta = custom_fbeta(y_test, y_pred_top3, beta)

#calculate rejection rate
n_rejections = np.sum(y_pred_top3==9)
rejection_rate = np.mean(y_pred_top3==9)
results_rejection_rate = rejection_rate

#calculate accuracy on not rejected samples
n_false = np.sum(y_pred_top3!=y_test) - n_rejections
acc = (len(y_pred_top3)-n_rejections-n_false)/(len(y_pred_top3)-n_rejections)
results_acc = acc

#Baseline: Achieves precision of xx% and Accuracy of 69% (for not rejected data)


#%%HIVE COTE2 reject option - Top 3
##############################################################################

#%%Prepare HIVE COTE2 reject option
#Define Standard HIVECOTE 2 and add threshold via proba funct

#Decision Func Probas
def decision_func_probability(probas, threshold):
    
    #Delete all non-maximal values of a column
    #First create boolean mask from max values
    max_val_bool = np.zeros(probas.shape, dtype=bool) 
    max_val_bool[np.arange(len(probas)), probas.argmax(axis=1)] = 1
    #Select only those max values from the input array keeping shape constant
    max_vals = probas*max_val_bool
    
    #compare probability to threshold and prioritize with multiplication
    mask = 2*(max_vals > threshold)
    
    #extend array to include potential rejection option
    ext_mask = np.pad(mask,((0,0),(0,1)),mode='constant',constant_values=1) #shape=(#point, #classes)
    
    return ext_mask.argmax(axis=1) #shape=(#point, 1) 



class HIVECOTE2rejectOption_proba(HIVECOTEV2):
    
   def __init__(
        self,
        stc_params={
        "estimator": RotationForest(n_estimators=3),
        "n_shapelet_samples": 500, 
        "max_shapelets": 20,
        "batch_size": 100,
        },
        drcif_params={"n_estimators": 10},
        arsenal_params={"num_kernels": 100, "n_estimators": 5},
        tde_params={
        "n_parameter_samples": 25,
        "max_ensemble_size": 5,
        "randomly_selected_params": 10,
        },
        time_limit_in_minutes=0,
        save_component_probas=True,
        verbose=1,
        n_jobs=1,
        random_state=123,
        threshold=0.4
    ):
        self.stc_params = stc_params
        self.drcif_params = drcif_params
        self.arsenal_params = arsenal_params
        self.tde_params = tde_params

        self.time_limit_in_minutes = time_limit_in_minutes

        self.save_component_probas = save_component_probas
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.stc_weight_ = 0
        self.drcif_weight_ = 0
        self.arsenal_weight_ = 0
        self.tde_weight_ = 0
        self.component_probas = {}

        self._stc_params = stc_params
        self._drcif_params = drcif_params
        self._arsenal_params = arsenal_params
        self._tde_params = tde_params
        self._stc = None
        self._drcif = None
        self._arsenal = None
        self._tde = None
        self.threshold = threshold

        super(HIVECOTEV2, self).__init__()

        # We need to add is-fitted state when inheriting from scikit-learn
        self._is_fitted = False
        
   def predict(self, X):
        yPred = decision_func_probability(self.predict_proba(X),self.threshold)   
        return yPred

#%% Define Grid for nested kCV
#Extensive params grid
param_grid= {'stc_params': ({
        "estimator": RotationForest(n_estimators=3),
        "n_shapelet_samples": 500, 
        "max_shapelets": 20,
        "batch_size": 100,
    },
    {
        "estimator": RotationForest(n_estimators=2),
        "n_shapelet_samples": 100, 
        "max_shapelets": 5,
        "batch_size": 50,
    }),
    'drcif_params': ({"n_estimators": 10}, 
                     {"n_estimators": 5}),
    'arsenal_params': ({"num_kernels": 200, "n_estimators": 10},
                       {"num_kernels": 50, "n_estimators": 2}),
    'tde_params': ({
        "n_parameter_samples": 25,
        "max_ensemble_size": 5
    },
    {
        "n_parameter_samples": 50,
        "max_ensemble_size": 10
    },
    {
        "n_parameter_samples": 12,
        "max_ensemble_size": 3,
        "randomly_selected_params": 5,
    }),
    'threshold': [0.2, 0.5, 0.8]    
    }


#%%HIVE COTE2 reject option - Top 3

#generate random integers to fix random_seed for each model in the inner loop of the nested CV
beta = 0.5
n_outer_splits = 3
n_inner_splits = 3
rng = np.random.default_rng(seed=123)
hivecote_random_seeds = rng.integers(low=0, high=1000000, size = n_outer_splits)

#------------------------Nested-CV HIVECOTE V2-----------------------------------------------
print("Results for NCV for HIVECOTE2 with entropy threshold")
#reset iteration variable i
i = 0
#create individual scorer
fpoint5_scorer = make_scorer(custom_fbeta, beta=beta)
# configure the cross-validation procedure
cv_outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)

# define lists
outer_results = list()
results_fbeta = list()
results_acc = list ()
results_rejection_rate = list()


for train_ix, test_ix in cv_outer.split(X,y):
    # split data
    X_train, X_test = all_data['X'][train_ix], all_data['X'][test_ix]
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train, y_test = all_data['y'][train_ix], all_data['y'][test_ix]
    # configure the cross-validation procedure
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
    # define the model
    model = HIVECOTE2rejectOption_proba(random_state=hivecote_random_seeds[i])
    # search space defined above
    # define search
    search = GridSearchCV(model, param_grid, scoring=fpoint5_scorer, cv=cv_inner, refit=True)
    # execute search
    result = search.fit(X_train, y_train)
    # get the best performing model fit on the whole training set
    best_model = result.best_estimator_
    yhat = best_model.predict(X_test)
    
    #get top3 prediction
    yhat_proba = best_model.predict_proba(X_test)
    yhat_top3_type = np.argsort(yhat_proba, axis=1)[:,::-1][:,:3]
        
    #for cases where algorithm rejects device: prob value -> 0, type -> 9
    rej = (yhat != 9)
    yhat_top3_type[np.invert(rej)] = 9
    
    #compare test labels with top3 predictions
    compare_matrix = np.tile(np.array(y_test)[:,None],(1,3)) == yhat_top3_type
    correct_prediction_mask = np.amax(compare_matrix, axis=1, keepdims = False)
    
    #put together a matrix to feed it into the usual scorer
    #take all in the Top3 included labels from y_test, for all other entries just take the wrong predicted class or rejection
    y_pred_top3 = (y_test*correct_prediction_mask) + (yhat*np.invert(correct_prediction_mask))
    results_fbeta.append(custom_fbeta(y_test, y_pred_top3, beta))
    
    #calculate rejection rate
    n_rejections = np.sum(y_pred_top3==9)
    rejection_rate = np.mean(y_pred_top3==9)
    results_rejection_rate.append(rejection_rate)
    
    #calculate accuracy on not rejected samples
    n_false = np.sum(y_pred_top3!=y_test) - n_rejections
    acc = (len(y_pred_top3)-n_rejections-n_false)/(len(y_pred_top3)-n_rejections)
    results_acc.append(acc)
    
    
    #store values from iteration (incl. best model) in dict
    #Initialize and Declare dict key    
    if (i == 0):
        result_dict_opt_model_multiclass = {i: best_model}
    else:
        result_dict_opt_model_multiclass.update({i: best_model})
    
    #report progress
    print(i)
    #iterate
    i += 1

#Averaged acc over all three runs: 85,2%

#%%Execute Repeated CV with best model
#For the final repeated kCV, the the first model from the nested kCV was chosen. It achieves a medium 
#accuracy on the test set but has rejected far less testcases than the third
#model.

#%% Define HC Model for subsequent repeated cross validation

hc2rej = HIVECOTE2rejectOption_proba(
    stc_params={
        "estimator": RotationForest(n_estimators=3),
        "n_shapelet_samples": 500, 
        "max_shapelets": 20,
        "batch_size": 100,
    },
    drcif_params={"n_estimators": 5},
    arsenal_params={"num_kernels": 200, "n_estimators": 10},
    tde_params={
        "n_parameter_samples": 25,
        "max_ensemble_size": 5,
        "randomly_selected_params": 5
    },
    threshold=0.5,
    random_state=1,
    time_limit_in_minutes=5
    )


#%% Define function to build model and calc performance metrics
def measure_KPIs(classifier, X_train, y_train, X_test, y_test):
    
    #train classifier and predict
    classifier.fit(X_train, y_train)
    yhat = classifier.predict(X_test)
    
    #get top3 prediction
    yhat_proba = classifier.predict_proba(X_test)
    yhat_top3_type = np.argsort(yhat_proba, axis=1)[:,::-1][:,:3]
        
    #for cases where algorithm rejects device: prob value -> 0, type -> 9
    rej = (yhat != 9)
    yhat_top3_type[np.invert(rej)] = 9
    
    #compare test labels with top3 predictions
    compare_matrix = np.tile(np.array(y_test)[:,None],(1,3)) == yhat_top3_type
    correct_prediction_mask = np.amax(compare_matrix, axis=1, keepdims = False)
    
    #put together a matrix to feed it into the usual scorer
    #take all in the Top3 included labels from y_test, for all other entries just take the wrong predicted class or rejection
    y_pred_top3 = (y_test*correct_prediction_mask) + (yhat*np.invert(correct_prediction_mask))
    fbeta = custom_fbeta(y_test, y_pred_top3, beta)
    
    #calculate rejection rate
    n_rejections = np.sum(y_pred_top3==9)
    rejection_rate = np.mean(y_pred_top3==9)
    
    #calculate accuracy on not rejected samples
    n_false = np.sum(y_pred_top3!=y_test) - n_rejections
    acc = (len(y_pred_top3)-n_rejections-n_false)/(len(y_pred_top3)-n_rejections)
    
    return acc, y_test, yhat, yhat_proba, y_pred_top3, fbeta, rejection_rate, classifier

#%% Conduct repeated kCV
n_splits = 3
n_repeats = 3
number_of_algorithms = 1

"""
acc, y_test, yhat, yhat_proba, y_pred_top3, fbeta, rejection_rate, classifier
"""

res_acc = []
res_y_test = []
res_yhat = []
res_yhat_proba = []
res_y_pred_top3 = []
res_fbeta = []
res_rejection_rate = []
res_classifier = []


rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state = 1)

#produce series of random but reproducible seeds to use during loops
rng = np.random.default_rng(seed=1)
loop_random_seeds = rng.integers(low=0, high=1000000, size = n_splits*n_repeats*number_of_algorithms)


#Define loop for repeated kCV
i = 0

for train_index, test_index in rskf.split(X, y):

    #Define storage location for split results
    results_table = []
    
    #Define data for split
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #Convert to DataFrame
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    
    
    #Train model and calc metrics
    acc, y_test, yhat, yhat_proba, y_pred_top3, fbeta, rejection_rate, classifier = measure_KPIs(hc2rej, X_train, y_train, X_test, y_test)
    
    res_acc.append(acc)
    res_y_test.append(y_test)
    res_yhat.append(yhat)
    res_yhat_proba.append(yhat_proba)
    res_y_pred_top3.append(y_pred_top3)
    res_fbeta.append(fbeta)
    res_rejection_rate.append(rejection_rate)
    res_classifier.append(classifier)
    
    i = i+1

#------End of for-loop--------
#Model achieves on average over all nine test runs of the 3x3 kCV 83,73% accuracy 

#%%Plot exemplary confusion matrix
#For the exemplary confusion matrix, test run 8 was chosen with an acc of 81%

#Rename lists for cm
y_test_cm = res_y_test[1]
y_pred_cm = res_y_pred_top3[1]

#Calc cm
cm_hivecote_top3 = confusion_matrix(y_test_cm, y_pred_cm)

#Plot cm
#define color code for plot
low_color = "#DAE8FC"
high_color = "#6C8EBF"

#create color map
cmp_cstm = LinearSegmentedColormap.from_list(
    "cstm",[low_color, high_color], N=6)


sn.set(font='serif')
sn.heatmap(cm_hivecote_top3, annot=True, cmap=cmp_cstm, linewidth=.5,
            xticklabels=['Vibration', 'Position', 'Spannung', 'Leistung', 
                        'Stromstärke', 'Geschwindigkeit', 'Beschleunigung',
                        'Temperatur', 'Gyroscop', 'keine Vorhersage'], 
           yticklabels=['Vibration', 'Position', 'Spannung', 'Leistung', 
                        'Stromstärke', 'Geschwindigkeit', 'Beschleunigung',
                        'Temperatur', 'Gyroscop', 'keine Vorhersage'])

plt.xticks(rotation=45, ha='right')
plt.xlabel("Vorhersage", labelpad=10)
plt.ylabel("Wahre Geräteklasse", labelpad=10)



plt.savefig('C:/Users/USER/OneDrive - Technische Universität Berlin/Desktop/D/Mongrafie/Grafiken/FinaleGrafiken/6.4/Finale_Konfusionsmatrix_ZentrGID.svg', format='svg')

plt.show()

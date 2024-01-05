import numpy as np
from scipy.stats import entropy


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, fbeta_score, confusion_matrix



def custom_f1(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average = 'micro', labels = np.unique(y_true))
    return f1

def custom_fbeta(y_true, y_pred, beta):
    f1 = fbeta_score(y_true, y_pred, average = 'micro', beta=beta, labels = np.unique(y_true))
    return f1

#funktionsweise implementieren was passiert wenn same probability
#implementiere weiteren parameter_ distanz, der den Abstand zum zweit wahrscheinlichsten optimiert

def decision_func_entropy(probas, threshold):
    
    #Create boolean mask from max values - all zeros but 1 where max_value is located
    max_val_bool = np.zeros(probas.shape, dtype=bool) 
    max_val_bool[np.arange(len(probas)), probas.argmax(axis=1)] = 1
           
    #calculate columnwise entropy of all entries
    entr = entropy(probas,axis=1, base=2)/np.log2(probas.shape[1])
    entr = entr > threshold
    entr=entr[:,None]
    
    #extend array to include potential rejection option if entropy is higher than threshold (*2 for prioritization)
    ext_matrix = np.concatenate((max_val_bool,entr*2),axis=1)
    
    return ext_matrix.argmax(axis=1) #shape=(#point, 1) 

def decision_func_rel_entropy(probas, threshold, class_distribution):
    
    #Create boolean mask from max values - all zeros but 1 where max_value is located
    max_val_bool = np.zeros(probas.shape, dtype=bool) 
    max_val_bool[np.arange(len(probas)), probas.argmax(axis=1)] = 1
       
    class_distribution = np.tile(class_distribution,(probas.shape[0],1))
    
    #calculate columnwise entropy of all entries
    entr = entropy(probas, qk = class_distribution,axis=1, base=2)/np.log2(probas.shape[1])
    entr = entr > threshold
    entr=entr[:,None]
    
    #extend array to include potential rejection option if entropy is higher than threshold (*2 for prioritization)
    ext_matrix = np.concatenate((max_val_bool,entr*2),axis=1)
    
    return ext_matrix.argmax(axis=1) #shape=(#point, 1) 


def decision_func_probability(probas, threshold):
    
    #Delete all non-maximal values of a column
    #---First create boolean mask from max values
    max_val_bool = np.zeros(probas.shape, dtype=bool) 
    max_val_bool[np.arange(len(probas)), probas.argmax(axis=1)] = 1
    #---Select only those max values from the input array keeping shape constant
    max_vals = probas*max_val_bool
    
    #compare probability to threshold and prioritize with multiplication
    mask = 2*(max_vals > threshold)
    
    #extend array to include potential rejection option
    ext_mask = np.pad(mask,((0,0),(0,1)),mode='constant',constant_values=1) #shape=(#point, #classes)
    
    return ext_mask.argmax(axis=1) #shape=(#point, 1) 
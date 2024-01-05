import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#new build_sktime_data
def build_sktime_data(sensor_dic, ts_length):
    #Iteratively putting together a list and then adding it to a DataFrame is computationally less expensive than iteratively building a DataFrame
    #Source: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html
    
    all_data = pd.DataFrame(columns=['X','Y'])
        
    #ensure the biggest class 0 and label classes in descending order of their size
    #this is done by iterating through all DataSets because each DataFrame might contain different amounts of DataSets
    data_count = []
    
    for sensor_type in sensor_dic.keys(): #iteration over all sensor types
        count = 0 #count number of datasets per task
        for key in sensor_dic[sensor_type].keys(): #--------------iteration over all data sources for a particular sensor type
            dataset = pd.DataFrame(sensor_dic[sensor_type][key]) #[:desired_timeseries_length]
            for sensor in dataset:
                count += 1
        data_count.append([sensor_type,count]) #when all datasets have been counted add name of the current sensor_type and count to list
    
    #build DataFrame to simplify sorting            
    data_count = pd.DataFrame(data_count, columns = ['sensor_type', 'quantity'])
    keys_sorted = data_count.sort_values(by='quantity',ascending=False).iloc[:,0]

    #list to build data iteratively
    data_list=[]
    
    for classlabel, sensor_type in enumerate(keys_sorted): #iteration over all sensor types
        for key in sensor_dic[sensor_type].keys(): #--------------iteration over all data sources for a particular sensor type
            dataset = pd.DataFrame(sensor_dic[sensor_type][key])[:ts_length] #[:desired_timeseries_length]
            for sensor in dataset: # iteration over all columns of a dataframe, pd.DataFrame() makes sure that even the one column files can be treated as DataFrames
                sensor_name = key + ':' + sensor           
                #new_row = [row_label, specific column of a df -> pd.series, label of the type of sensor, dimensions of dataset]
                new_row = [sensor_name,dataset[sensor],classlabel,len(dataset)]
                data_list.append(new_row)
                #print(sensor_type,label,key,sensor,pd.DataFrame(sensor_dic[sensor_type][key])[sensor])  
                
    all_data = pd.DataFrame(data = data_list, columns = ['dataset','X','y','series_length'])
    all_data = all_data.set_index('dataset')
    
    return keys_sorted, all_data




"""
old build_sktime_data
def build_sktime_data(sensor_dic):
    #Iteratively putting together a list and then adding it to a DataFrame is computationally less expensive than iteratively building a DataFrame
    #Source: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html
    
    all_data = pd.DataFrame(columns=['X','Y'])
    ts_length = 1000
    
    data_list=[]
    
    for classlabel, sensor_type in enumerate(sensor_dic.keys()): #iteration over all sensor types
        for key in sensor_dic[sensor_type].keys(): #--------------iteration over all data sources for a particular sensor type
            dataset = pd.DataFrame(sensor_dic[sensor_type][key])[:ts_length] #[:desired_timeseries_length]
            for sensor in dataset: # iteration over all columns of a dataframe, pd.DataFrame() makes sure that even the one column files can be treated as DataFrames
                sensor_name = key + '_' + sensor           
                #new_row = [row_label, specific column of a df -> pd.series, label of the type of sensor, dimensions of dataset]
                new_row = [sensor_name,dataset[sensor],classlabel,len(dataset)]
                data_list.append(new_row)
                #print(sensor_type,label,key,sensor,pd.DataFrame(sensor_dic[sensor_type][key])[sensor])  
                
    all_data = pd.DataFrame(data = data_list, columns = ['dataset','X','y','series_length'])
    #all_data = all_data.set_index('dataset')
    
    return all_data
"""

def sktime_to_tsfresh_converter(all_data):
    
    #-------------Convert X data into long format------
    arr = np.zeros((1,2)) #base numpy array
    nb_sensors = all_data.shape[0] #save number of iterations necessary
    print(nb_sensors)
    
    #id lookup implementieren
    
    for idx in range(nb_sensors):
        ts = all_data.iloc[idx,0]
        ts_arr = ts.to_numpy()[:,None]
        ts_arr = np.pad(ts_arr,((0,0),(1,0)),mode='constant', constant_values=idx)
        arr = np.concatenate((arr,ts_arr),axis=0)
    
    X_long = pd.DataFrame(arr[1:,:], columns=['id','value'])
    
    return X_long


def build_tsfresh_data():
    x = 1 
    return x
    
    
def data_stats(classes,y_train,y_test=[0]):
   
    if len(y_test)==1:
        labels, count = np.unique(y_train, return_counts=True)
        data_stats = pd.DataFrame(zip(classes,labels,count))
        data_stats.columns = ["sensor_type", "y_label_representation", "count_total"]
    else:
        labels, count_train = np.unique(y_train, return_counts=True)
        _ ,count_test = np.unique(y_test, return_counts=True)
        data_stats = pd.DataFrame(zip(classes,labels,count_train,count_test))
        data_stats.columns = ["sensor_type", "y_label_representation", "count_train", "count_test"]
    return data_stats

def build_nn_data(sensor_dic, ts_length):
    arr = np.empty((1,ts_length+1))
    
    #Built Numpy Array from data
    for classlabel, sensor_type in enumerate(sensor_dic.keys()): #iteration over all sensor types
        for key in sensor_dic[sensor_type].keys():#--------------iteration over all data sources for a particular sensor type
            dataset = pd.DataFrame(sensor_dic[sensor_type][key])[:ts_length]#potentially slice
            new_rows = dataset.to_numpy().T
            new_rows = np.pad(new_rows, ((0,0),(0,1)), mode='constant', constant_values = classlabel)
            arr = np.concatenate((arr,new_rows),axis=0)
    
    #Omit first row created by np.empty()
    all_data_array = arr[1:,:]            
    #print(all_data_array.shape)
    
    #Split into X and y data
    X = all_data_array[:,:ts_length]
    y = all_data_array[:,ts_length]
    
    return X,y

def calc_accuracy(acc_table, model, X_test, y_test):
    
    preds = model.predict(X_test) #Predict labels for test-set
    cm = confusion_matrix(y_test, preds) #Compute confusion matrix
    class_acc = cm.diagonal()/cm.sum(axis=1) #Normalize entries and use diagonal entries as per class accuracys
    
    model_name = str(model)
    acc_table[model_name] = class_acc #add accuracys to table
        
    acc = cm.diagonal().sum()/len(y_test) #compute total accuracy
    print(f"Accuracy on TestSet: {acc:.3f}")
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
        
    return acc_table

    
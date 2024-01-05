#%%Import Packages
import glob
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt


#%% Read in Data

#Read in Data 

def read_in_data(path_to_files, dtype_in, a_check,datatype):
    data = pd.DataFrame()
    path = path_to_files

    
    if (a_check == 0):
        
        all_files = glob.glob(path + "/*.txt")
        #Read in data with defined dtype for comp costs
        data = pd.concat((pd.read_csv(f, sep='\t', dtype=dtype_in) for f in all_files), ignore_index=True)
        #Fill NAs with 0 and delete all-zero columns
        data = data.fillna(0)
        data = data.loc[:, (data != 0).any(axis=0)]
        
        #Assumption: How to handle non-numeric values: If 10 or less distinct values: numerize else ignore for now
        #Declare del_list to remove not mappable columns
        del_list = []
        
        for column in data:
            #Attention: Currently special case for ID
            if (data[column].dtype != 'float') and (data[column].dtype != 'int') and (column != 'ID'):
                if data[column].nunique() < 10:
                    #convert unqiue values to numbers
                    #1. create dict of distinct values
                    remap_dict = dict([(y,x+1) for x,y in enumerate(set(data[column]))])
                    #2. replace non int and floats via dict values
                    data[column] = data[column].map(remap_dict) 
                else:
                    del_list.append(column)
        
        #Drop columns with too many non-numerical distinct values
        data = data.drop(columns = del_list)
    
    if (a_check == 1):
        
        all_files = glob.glob(path + "/*.csv")
        data = pd.concat((pd.read_csv(f, sep=';', dtype=dtype_in) for f in all_files), ignore_index=True)

        #Fill NAs with 0 and delete all-zero columns
        data = data.fillna(0)
        data = data.loc[:, (data != 0).any(axis=0)]

        drop_list = ['K', 'L', 'R', 'M', 'B', 'ID', 'BZ', 'M_P', 'M_G', 'W_P', 'W_G', 'A', 'MX', 'NR', 'D']
        data = data.drop(columns = drop_list)


        del_list = []

        for column in data:
            #Attention: Currently special case for ID
            if (data[column].dtype != 'float') and (data[column].dtype != 'int') and (column != 'ID1') and (column != 'ID2'):
                #convert unqiue values to numbers
                #1. create dict of distinct values
                remap_dict = dict([(y,x+1) for x,y in enumerate(set(data[column]))])
                #2. replace non int and floats via dict values
                data[column] = data[column].map(remap_dict)
    
        data = data[data.S == 'X']
    
    return data

#%% Data handling - Bring data into dict format

'''
1. Create Dictionary with keys = diffferent attributes
2. Dictionary is sorted by different devices
3. Timeseries of different devices are brought into same length
'''

def handling_data(data, series_length, ID):


    data_dict = {}
    
    #Go through the enl df to 1. bring timeseries to defined length, 2. store in dict
    for column in data.drop([ID],axis=1):
     
        #Create new dict entry - column equals feature in dataset
        data_dict.update({column:{}})
     
        data_feature_df = data.loc[:, [ID, column]]
        data_feature_dict = {k: v for k, v in data_feature_df.groupby(ID)}
        data_feature_dict = {k: v for k, v in data_feature_dict.items() if v.shape[0] > 100} # Check if line can be included above
            
        #1. Loop: Go through the individual tools in the dict
        for key, df in data_feature_dict.items():
            
            # a is the timeseries of a given feature (column) of a given device (key)
            a = data_feature_dict[key]
            a = a[column]
        
                
            #strech/cut ts length
            if a.shape[0] >= series_length:
                a_new = a.iloc[:series_length,]
                a_new = pd.DataFrame(a_new)
                a_new.columns = [key]
                a_new.columns = a_new.columns.astype(str) #added
    
            else:
                values = a.to_numpy()
                np_plus = np.tile(values,(int(series_length/values.shape[0]),))
                rest = series_length - np_plus.shape[0]
                np_plus = np.concatenate((np_plus,values[:rest,]))
                a_new = pd.DataFrame(np_plus,columns = [key])
                a_new.columns = a_new.columns.astype(str) #added
    
            
            key_str = str(key) #Required fpr building sktime_data
            data_dict[column][key_str] = a_new

    return data_dict


#%% Verify prediction results

#Map y_test to plaintext labels
def map_to_plaintext_labels(preds, y_test, tt_data_stats, true_label, predicted_label):
    
    y_test_plaintext = pd.merge(y_test, tt_data_stats, how='left', left_on=['y'], right_on=['y_label_representation'])
    y_test_plaintext = y_test_plaintext.drop(['y_label_representation','count_train','count_test'],1)
    y_test_plaintext = y_test_plaintext.rename(columns={'sensor_type': true_label})
    
    #Map pred to plaintext labels
    preds_plaintext = pd.merge(pd.DataFrame(preds), tt_data_stats, how='left', left_on=[0], right_on=['y_label_representation'])
    preds_plaintext = preds_plaintext.rename(columns={0: 'predicted_label','sensor_type': predicted_label})
    preds_plaintext = preds_plaintext.drop(['y_label_representation','count_train','count_test'],1)
    
    #Evaluate Overall Prediction results
    prediction_results = pd.concat([y_test_plaintext, preds_plaintext], axis = 1)
    count_series = prediction_results.groupby([true_label, predicted_label]).size()
    count_series = pd.DataFrame(count_series)
    count_series = count_series.rename(columns={0: 'Predictions'})
    
    #Plot Confusion Matrix
    tessst = count_series.unstack(fill_value=0)
    svm = sn.heatmap(tessst, annot=True,cmap='coolwarm', linecolor='white', linewidths=1,xticklabels=1, yticklabels=1)
    svm.set_xticklabels(svm.get_xticklabels())


    # get the tick label font size
    fontsize_pt = 12
    dpi = 72.27
    
    # comput the matrix height in points and inches
    matrix_height_pt = fontsize_pt * tessst.shape[0]
    matrix_height_in = matrix_height_pt / dpi
    
    # compute the required figure height 
    top_margin = 0.04  # in percentage of the figure height
    bottom_margin = 0.04 # in percentage of the figure height
    figure_height = matrix_height_in / (1 - top_margin - bottom_margin)
    
    
    # build the figure instance with the desired height
    fig, ax = plt.subplots(
            figsize=(12,figure_height), 
            gridspec_kw=dict(top=1-top_margin, bottom=bottom_margin))
    
    # let seaborn do it's thing
    ax = sn.heatmap(tessst, ax=ax, linecolor='white',cmap='coolwarm', linewidths=1,xticklabels=1, yticklabels=1)
    
    
    # save the figure
    figure = ax.get_figure()
    plt.setp(ax.get_xticklabels(), 'rotation', 45, ha='right')
    figure.savefig('new_plot.png', dpi=1000, bbox_inches='tight')
    plt.close()


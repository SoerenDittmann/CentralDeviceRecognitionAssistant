'''
Script for the experimental evaluation of "Semantische Assistenzen für Digitale Zwillinge"
Script to:
    1. Store overview of databases
    2. Monitor available data in databases
    3. Detect new data
    4. Classify new data
    5. Send notifications and top results to frontend

Please note: The script below is written iterable that new data bases can be added
(to the overviewdatabases dict) without changing the script much. Please check
necessary iterations especially over the dict when new data sources are added.
'''
#%%Import packages
import io
import sys
import os
import pickle
import copy
import base64
import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import json
from pathlib import Path
import csv
from itertools import zip_longest


#Packages for classification
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.sklearn._rotation_forest import RotationForest
from sktime.datatypes._check import check_is_mtype, mtype
from Classification.custom_classifiers.classifiers import custom_fbeta 
from Classification.custom_classifiers.utils import build_sktime_data, calc_accuracy, data_stats
from Classification.data_handling.basics import read_in_data, handling_data, map_to_plaintext_labels

#%% 1. Store overview of databases

'''
The overviewdatabases dict serves as an overview about the data storages within
the data persistent layer of the virtual component of the complex digital twin.
New data storages can be added as keys to the dict and details be stored as 
indicated below.
'''

#Define dict with an overview of the databases
#Necessity to distiguish between databases and object stores due to their 
#different "APIs"
overviewdatabases = {}
overviewobjstores = {}


#Define dict with an overview of known data in the virtual component
knownData_DB = {}
knownData_objstores = []
#Define dict of found data for diff with knowndata
foundData_DB = {}
foundData_objstores = []
counter_files = 0

#Define dict of new data as diff between known and found data
newData = {}
newlyFoundData = []
new_files = []
series_length = 1000


#Assign database details to dict 
overviewdatabases['DigitalTwin_database']={}
overviewdatabases['DigitalTwin_database']['Connection_details']={
    'POSTGRES_ADDRESS':'127.0.0.1', ## INSERT YOUR DB ADDRESS IF IT'S NOT ON PANOPLY
    'POSTGRES_PORT': '5432',
    'POSTGRES_USERNAME': '', ## CHANGE THIS TO YOUR PANOPLY/POSTGRES USERNAME
    'POSTGRES_PASSWORD': '', ## CHANGE THIS TO YOUR PANOPLY/POSTGRES PASSWORD
    'POSTGRES_DBNAME':'DigitalTwin_database'} ## CHANGE THIS TO YOUR DATABASE NAME

#Define connection string
postgres_str = ('postgresql://{username}:{password}@{ipaddress}:{port}/{dbname}'
.format(username=overviewdatabases['DigitalTwin_database']['Connection_details']['POSTGRES_USERNAME'],
password=overviewdatabases['DigitalTwin_database']['Connection_details']['POSTGRES_PASSWORD'],
ipaddress=overviewdatabases['DigitalTwin_database']['Connection_details']['POSTGRES_ADDRESS'],
port=overviewdatabases['DigitalTwin_database']['Connection_details']['POSTGRES_PORT'],
dbname=overviewdatabases['DigitalTwin_database']['Connection_details']['POSTGRES_DBNAME']))


#Store connection string for later retreval in dict
overviewdatabases['DigitalTwin_database']['Connection_string']={
   'CONNECTION_STRING': postgres_str}

#Define connection itself
cnx = create_engine(overviewdatabases['DigitalTwin_database']['Connection_string']['CONNECTION_STRING'])


#Store connection itself in dict for easier acces later
overviewdatabases['DigitalTwin_database']['Connection']={
    'CONNECTION': cnx}

#Assign object oriented storage to dict
overviewobjstores['objectStore']={}

objectStore_str = "C:/Users/USER/.../CentralDeviceRecognitionAssistant_objectStore/data/"

#Store connection string for later retreval in dict
overviewobjstores['objectStore']['Connection_string']={'CONNECTION_STRING': objectStore_str}




#%% 2. Monitor available data in the data sources of the virtual component
'''
Monitoring the avalable data consists of the substeps:
    1. Connect to data storages
    2. If database: Query a list of all schemas within the data storage
        Differentiate between organisational schemas and "data" schemas
    3. If database: Query a list of all tables within the data storage
    4. If database: Query a list of all columns within the data storage
        Assumption: New data in a timeseries database results in a new column/
                    new columns in a new table/new schema
    5. Store list of known data internally

Please note: The solution simplifies to the extend that the server is not
also queried for all databases before step 2.
'''

#----
#----A: For databases
#1. Connect to data storages
cnx = overviewdatabases['DigitalTwin_database']['Connection']['CONNECTION']

#2. Query a list of all schemas within the data storage
query_schemas = 'select schema_name from information_schema.schemata'
#Save query result in df
schemas = pd.read_sql_query(query_schemas, con=cnx)
#Define orga schemas
orga_schemas = schemas.loc[1:,:]
#Differentiate between organisational schemas and "data" schemas
schemas = schemas[~schemas.isin(orga_schemas)].dropna()
#Save schemas to query in dict
overviewdatabases['DigitalTwin_database']['RelevantSchemas']=schemas

#3. Query a list of all tables within the schemas in the data storage
query_tables = 'select * from information_schema.tables where table_schema = \'{schema}\''.format(schema = overviewdatabases['DigitalTwin_database']['RelevantSchemas']['schema_name'][0])
tables = pd.read_sql_query(query_tables, con=cnx)
#Save tables to query in dict
overviewdatabases['DigitalTwin_database']['RelevantTables']=tables

#4. Query a list of all columns within all tables within the data storage
for databases in overviewdatabases:
    database = '{databsename}'.format(databsename=databases)
    foundData_DB[database]={}
    for schema in overviewdatabases[database]['RelevantSchemas']['schema_name']:
        schema = '{schemaname}'.format(schemaname=schema)
        foundData_DB[database][schema]={}
        for tables in overviewdatabases[database]['RelevantTables']['table_name']:
            query_columns = 'SELECT column_name FROM information_schema.columns WHERE table_schema = \'{schema}\' AND table_name = \'{table}\';'.format(schema = overviewdatabases['DigitalTwin_database']['RelevantSchemas']['schema_name'][0],table = tables)
            key = '{tablename}'.format(tablename=tables)
            foundData_DB[database][schema][key] = pd.read_sql_query(query_columns, con=cnx)

#----B: For object oriented storages:
'''
Get list of all files. If a file is new, read in file and classify data
Assumption 1: Directory only contains .txt or .csv files. 
Code however only tested for csv files.
Assumption 2: Data structure within a given file is constant. New data always
means new file.
Assumption 3: Imported files can contain header or not. Therefore the first
ten entries are disregarded. The 11th entry is used to differentiate known data
from new data.
'''
#Helper function to automatically determine the separator
#Source: https://stackoverflow.com/questions/69817054/python-detection-of-delimiter-separator-in-a-csv-file
#acc: 13.12.2023
def find_delimiter(filename):
    sniffer = csv.Sniffer()
    with open(filename) as fp:
        delimiter = sniffer.sniff(fp.read(400)).delimiter
    return delimiter




for stores in overviewobjstores:
    store = '{storename}'.format(storename=stores)
    for repos in overviewobjstores[store]:
        repo = '{reponame}'.format(reponame=repos)
        
        for files in os.listdir(overviewobjstores[store][repo]['CONNECTION_STRING']):
            file_path = os.path.join(overviewobjstores[store][repo]['CONNECTION_STRING'], files)
            
            #Read in the individual files to determine the columns in each file
            #required differentiation between .txt and .csv
          
            #get file extension to handle csv and txt
            file_extension = Path(file_path).suffix
            
            if file_extension == ".csv":
                sep = find_delimiter(file_path)
                #seperator deliberately left variable, header always disregarded
                csv_df = pd.read_csv(file_path, header=None, sep=sep)
                #to account for different structrures disregard first 10 rows
                csv_df = csv_df.iloc[10:,:]
                
                #store nbr of columns and first entry in found data
                nbr_columns_found = len(csv_df.columns)
                column_titles_found = csv_df.iloc[0].values
                
            if file_extension == ".txt":
                #seperator deliberately left variable, header always disregarded
                csv_df = pd.read_csv(file_path, header=None)
                #to account for different structrures disregard first 10 rows
                csv_df = csv_df.iloc[10:,:]
                
                #store nbr of columns and first entry in found data
                nbr_columns_found = len(csv_df.columns)
                column_titles_found = csv_df.iloc[0].values
            
            newlyFoundData = [counter_files, files, file_path, nbr_columns_found, column_titles_found]
            counter_files += 1
            #newlyFoundData = [counter_files, files, file_path, 2, [112.35,131]]
            foundData_objstores.append(newlyFoundData)
            
            



#%%3. Detect new data
'''
Iterate trough found data and check if data is already known. If not,
the database, tablename, and/or columnname are stored in the newData dict.
Loops below iterate through every stage of the foundData dict and compares each
stage with the already known data.
'''
#%%Read known Data
try:
    with open('C:/Users/USER/.../knownData.pkl', 'rb') as knowndata_file:
        knownData_DB = pickle.load(knowndata_file)
except FileNotFoundError:
    knownData_DB = []

#knownData['DigitalTwin_database']['DigitalTwin_data_from_phys_comp']['ims_bearing'] = knownData['DigitalTwin_database']['DigitalTwin_data_from_phys_comp']['ims_bearing'].drop(3)

try:
    with open('C:/Users/USER/.../foundData_objstores.pkl', 'rb') as knowndata_file_obj:
        knownData_objstores = pickle.load(knowndata_file_obj)
except FileNotFoundError:
    knownData_objstores = []

tables_counter = 0
column_counter = 0

#DB
for databases in foundData_DB:
    database='{databsename}'.format(databsename=databases)
    if databases in knownData_DB:
        for schemas in foundData_DB[database]:
            schema = '{schemaname}'.format(schemaname=schemas)
            if schemas in knownData_DB[database]:
                for tables in foundData_DB[database][schema]:
                    if tables in knownData_DB[database][schema]:
                        compare_found_known = pd.merge(foundData_DB[database][schema][tables], knownData_DB[database][schema][tables], how='left', on='column_name', indicator=True)
                        if any(compare_found_known['_merge'] == 'left_only'):
                            only_found = compare_found_known[compare_found_known['_merge'] == 'left_only']
                            only_found = only_found.drop('_merge', axis=1)
                            if (tables_counter == 0) and (column_counter == 0):
                                newData[database]={}
                                newData[database][schema] = {}
                                tables_counter = tables_counter+1
                                column_counter = column_counter+1
                            newData[database][schema][tables] = pd.DataFrame()
                            newData[database][schema][tables] = pd.concat([newData[database][schema][tables], only_found], ignore_index=True)
                    else:
                        if tables_counter == 0:
                            newData[database] = {}
                            newData[database][schema] = {}
                            tables_counter = tables_counter+1
                        newData[database][schema][tables] = foundData_DB[database][schema][tables]
            else:
                newData[database][schema] = foundData_DB[database][schema]
    else:
        newData[database] = foundData_DB[database]

#Detect new files or files with data not yet processed
for linesfound, linesknown in zip_longest(foundData_objstores, knownData_objstores, fillvalue=None):
    
    if linesfound is not None and linesknown is not None:
        #check if file is already present, just not completely processes
        if linesfound[0] == linesknown[0]:
            if type(linesknown[4]) == np.ndarray and type(linesknown[4]) == np.ndarray:
                unknowncolumns = [col for col in linesfound[4] if col not in linesknown[4]]
                if len(unknowncolumns) > 0:
                    newdataline = [linesknown[0], linesknown[1], linesknown[2], linesknown[3], unknowncolumns]
                    new_files.append(newdataline)
            elif type(linesknown[4]) == list and type(linesknown[4]) == list:
                unknowncolumns = [col for col in linesfound[4] if col not in linesknown[4]]
                if len(unknowncolumns) > 0:
                    newdataline = [linesknown[0], linesknown[1], linesknown[2], linesknown[3], unknowncolumns]
                    new_files.append(newdataline)
            else:
                unknowncolumns = [col for col in linesfound[4] if col != linesknown[4]]
                if len(unknowncolumns) > 0:
                    newdataline = [linesknown[0], linesknown[1], linesknown[2], linesknown[3], unknowncolumns]
                    new_files.append(newdataline)   
                
        #new file
        else:
            newdataline = linesfound
            new_files.append(newdataline)
    else:
        newdataline = linesfound
        new_files.append(newdataline)  
    
  
    
#%%4. Classify new data with HIVE COTE 2.0 model and export all information
'''
The classification requries the following substeps:
    1. Load trained model
    2. Load data from sources that is flaged in newData
    3. Map data into the format expected by the classifier
    4. predict and send results to frontend
    
'''
#%% Define model
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


#%%Load final classifier
#To execute the following lines, please make sure to load the trained multiclass
#classification  model into the selected working directory: Final_Multiclass_Model.pkl

with open('C:/Users/USER/.../Final_Multiclass_Model.pkl', 'rb') as model_file:
    hc2rejclass = pickle.load(model_file)


#%%Export newly found data incl prediction

if bool(newData) == True:

    for databases in newData:
        database='{databsename}'.format(databsename=databases)
        for schemas in newData[database]:
            schema = '{schemaname}'.format(schemaname=schemas)
            for tables in newData[database][schema]:
                table='{tablename}'.format(tablename=tables)
                for columns in newData[database][schema][table]['column_name']:
                    column='{columnname}'.format(columnname=columns)
                    
                    #Query data for plot
                    query_for_plot = 'SELECT {columnname} FROM \"{schemaname}\".\"{tablename}\"'.format(
                        columnname=column, schemaname=schema, tablename=table)

                    
                    df_for_plot = pd.read_sql_query(query_for_plot, con=cnx)
                    #Convert to series for plot, otherwise the plot fails
                    df_for_plot = df_for_plot.iloc[:,0]
                    df_for_plot = pd.to_numeric(df_for_plot)
                    
                    #Write predicted data to knowndata dataframe
                    #Check if knowndata is empty
                    '''
                    Please note: Below the pre-defined column name: column_name
                    is used, since this is the standard name, when querying 
                    available columns in a postgreSQL table.
                    '''
                    if bool(knownData_DB):
                        #check if schema is already known
                        if bool(knownData_DB[database][schema]):
                            #check if table with column names is defined
                            if len(knownData_DB[database][schema][table]) > 0:
                                new_row_db = pd.DataFrame({'column_name': column}, index=[0])
                                knownData_DB[database][schema][table] = pd.concat([knownData_DB[database][schema][table], 
                                                                                   new_row_db], ignore_index=True)
                            #if table with column names not yet defined, define it
                            else:
                                knownData_DB[database][schema][table] = pd.DataFrame(columns=['column_name'])
                                new_row_db = pd.DataFrame({'column_name': column}, index=[0])
                                knownData_DB[database][schema][table] = pd.concat([knownData_DB[database][schema][table], 
                                                                                   new_row_db], ignore_index=True)
                        #if schema not known, define new schema
                        else:
                            knownData_DB[database][schema]={}
                            knownData_DB[database][schema][table] = pd.DataFrame(columns=['column_name'])
                            new_row_db = pd.DataFrame({'column_name': column}, index=[0])
                            knownData_DB[database][schema][table] = pd.concat([knownData_DB[database][schema][table], 
                                                                               new_row_db], ignore_index=True)
                    #first entry to known data
                    else:
                        knownData_DB={}
                        knownData_DB[database]={}
                        knownData_DB[database][schema]={}
                        knownData_DB[database][schema][table] = pd.DataFrame(columns=['column_name'])
                        new_row_db = pd.DataFrame({'column_name': column}, index=[0])
                        knownData_DB[database][schema][table] = pd.concat([knownData_DB[database][schema][table], 
                                                                           new_row_db], ignore_index=True)
                        
                        
                    #Generate plot
                    s = io.BytesIO()
                    plt.plot(df_for_plot, linestyle='-', color="#5078b4")
                    plt.xlabel('Zeit []')
                    plt.ylabel('Wert []')
                    plt.grid(True)
                    plt.savefig(s, format='png',bbox_inches='tight')
                    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
                    
                    #Predict newly found data with predefined model
                    for columns in range(len(newData[database][schema][table])):   
                        query_newdata = 'SELECT {q_column} FROM \"{q_schema}\".{q_table}'.format(
                            q_column = newData[database][schema][table].iloc[columns,0],
                            q_schema = schema,
                            q_table = table)

                        newData_forpred = pd.read_sql_query(query_newdata, con=cnx, dtype=np.float64)
                    
                    newData_forpred = newData_forpred[:series_length]
                    newData_forpred_series = newData_forpred.squeeze(axis=1)

                    data_list=[]
                    newData_row = [newData_forpred_series]
                    data_list.append(newData_row)

                    formattest = pd.DataFrame(data_list, columns = ['X'])


                    yhat_proba = hc2rejclass.predict_proba(formattest)
                    yhat = hc2rejclass.predict(formattest)
                    yhat_top3_type = np.argsort(yhat_proba, axis=1)[:,::-1][:,:3]
                    #for cases where algorithm rejects device: prob value -> 0, type -> 9
                    rej = (yhat != 9)
                    yhat_top3_type[np.invert(rej)] = 9
                        
                    list_of_deviceclasses = ["vibration_sensor", "position_sensor",  "voltage_sensor", "power_sensor", 
                                 "current_sensor", "velocity_sensor", "acceleration_sensor",
                                 "temperature_sensor", "gyroscope", "no prediction"]

                    proba_inner = yhat_top3_type[0]

                    predictions_for_FE = [list_of_deviceclasses[deviceclass] for deviceclass in proba_inner]
                    
                    #Write knowndata file to machine
                    with open('C:/Users/USER/.../knownData.pkl', 'wb') as knowndata_file:
                        pickle.dump(knownData_DB, knowndata_file)
                    
                    #Export data for node-red stream
                    newData_found = {
                        'newData': True,
                        'database': database,
                        'schema': schema,
                        'table': table,
                        'image': '<img src="data:image/png;base64,%s" alt="Plot">' % s,
                        'prediction1': predictions_for_FE[0], 
                        'prediction2': predictions_for_FE[1],
                        'prediction3': predictions_for_FE[2] 
                        }
                    newData_exp = json.dumps(newData_found)
                    print(newData_exp)
                    sys.exit()


if bool(new_files) == True:
    for files in new_files:
                
        file_path = os.path.join(overviewobjstores[store][repo]['CONNECTION_STRING'], files[1])
        
        #Choose end of path to display in frontend
        p = Path(files[2])
        path_fe = p.parts[len(p.parts)-3:len(p.parts)-1]
        path_fe = path_fe[0] +"/" +path_fe[1]
        
        #create exemplary plot to display in Frotend
        #get file extension to handle csv and txt
        file_extension = Path(file_path).suffix

        
        if file_extension == ".csv":
            #determine separator with helper function
            sep = find_delimiter(file_path)
            #seperator deliberately left variable, header always disregarded
            csv_df = pd.read_csv(file_path, header=None, sep=sep)
            #to account for different structrures disregard first 10 rows
            csv_df = csv_df.iloc[10:,:]
            #choose so far unprocessed column from imported data
            csv_df = csv_df[csv_df.columns[csv_df.iloc[0] == files[4][0]][0]]
            
            #update known data
            if len(knownData_objstores) > 0:
                #Check if file is known
                count_rows_searched = 0
                for index, idx in enumerate(knownData_objstores):
                    count_rows_searched = count_rows_searched+1
                    if files[0] == [idx][0][0]:
                        #file is known
                        if type(files[4]) == list:
                            newcolumn = files[4][0]
                            if type([idx][0][4]) == list:
                                knownData_objstores[index][4].append(newcolumn)
                                break
                            else:
                                knownData_objstores[index] = [files[0], files[1], files[2], files[3], [[idx][0][4], newcolumn]]
                                #knownData_objstores[index] = updatedknownline
                                break
                        else:
                            if type([idx][0][4]) == list:
                                newcolumn = files[4]
                                knownData_objstores[index][4].append(newcolumn)
                                break
                            else:
                                newcolumn = files[4]
                                updatedknownline = [files[0], files[1], files[2], files[3], newcolumn]
                                knownData_objstores.append(updatedknownline)
                                break
                    
                    #file not known but entries in knownData
                    elif (files[0] != [idx][0][0]) and (count_rows_searched == len(knownData_objstores)):
                        if type(files[4]) == list:
                            newcolumn = files[4][0]
                            updatedknownline = [files[0], files[1], files[2], files[3], newcolumn]
                            knownData_objstores.append(updatedknownline)
                            break
                        elif type(files[4]) == np.ndarray:
                            newcolumn = files[4][0]
                            updatedknownline = [files[0], files[1], files[2], files[3], newcolumn]
                            knownData_objstores.append(updatedknownline)
                            break
                        else:
                            newcolumn = files[4]
                            updatedknownline = [files[0], files[1], files[2], files[3], newcolumn]
                            knownData_objstores.append(updatedknownline)
                            break
                
            #first entry in known data
            else:
                #write file to known data and set column to first column
                if type(files[4]) == list or type(files[4]) == np.ndarray:
                    newcolumn = files[4][0]
                    updatedknownline = [files[0], files[1], files[2], files[3], newcolumn]
                    knownData_objstores.append(updatedknownline)
                else:
                    newcolumn = files[4]
                    updatedknownline = [files[0], files[1], files[2], files[3], newcolumn]
                    knownData_objstores.append(updatedknownline)
                
        
            
        if file_extension == ".txt":
            #determine separator with helper function
            sep = find_delimiter(file_path)
            #seperator deliberately left variable, header always disregarded
            csv_df = pd.read_csv(file_path, header=None, sep=sep)
            #to account for different structrures disregard first 10 rows
            csv_df = csv_df.iloc[10:,:]
            #choose so far unprocessed column from imported data
            csv_df = csv_df[csv_df.columns[csv_df.iloc[0] == files[4][0]][0]]
            
            #update known data
            if len(knownData_objstores) > 0:
                #Check if file is known
                count_rows_searched = 0
                for index, idx in enumerate(knownData_objstores):
                    count_rows_searched = count_rows_searched+1
                    if files[0] == [idx][0][0]:
                        #file is known
                        if type(files[4]) == list:
                            newcolumn = files[4][0]
                            if type([idx][0][4]) == list:
                                knownData_objstores[index][4].append(newcolumn)
                                break
                            else:
                                knownData_objstores[index] = [files[0], files[1], files[2], files[3], [[idx][0][4], newcolumn]]
                                #knownData_objstores[index] = updatedknownline
                                break
                        else:
                            if type([idx][0][4]) == list:
                                newcolumn = files[4]
                                knownData_objstores[index][4].append(newcolumn)
                                break
                            else:
                                newcolumn = files[4]
                                updatedknownline = [files[0], files[1], files[2], files[3], newcolumn]
                                knownData_objstores.append(updatedknownline)
                                break
                    
                    #file not known but entries in knownData
                    elif (files[0] != [idx][0][0]) and (count_rows_searched == len(knownData_objstores)):
                        if type(files[4]) == list:
                            newcolumn = files[4][0]
                            updatedknownline = [files[0], files[1], files[2], files[3], newcolumn]
                            knownData_objstores.append(updatedknownline)
                            break
                        elif type(files[4]) == np.ndarray:
                            newcolumn = files[4][0]
                            updatedknownline = [files[0], files[1], files[2], files[3], newcolumn]
                            knownData_objstores.append(updatedknownline)
                            break
                        else:
                            newcolumn = files[4]
                            updatedknownline = [files[0], files[1], files[2], files[3], newcolumn]
                            knownData_objstores.append(updatedknownline)
                            break
                
            #first entry in known data
            else:
                #write file to known data and set column to first column
                if type(files[4]) == list or type(files[4]) == np.ndarray:
                    newcolumn = files[4][0]
                    updatedknownline = [files[0], files[1], files[2], files[3], newcolumn]
                    knownData_objstores.append(updatedknownline)
                else:
                    newcolumn = files[4]
                    updatedknownline = [files[0], files[1], files[2], files[3], newcolumn]
                    knownData_objstores.append(updatedknownline)
            
        
        #Convert to series for plot, otherwise the plot fails
        csv_df_for_plot = csv_df
        csv_df_for_plot = pd.to_numeric(csv_df_for_plot)
        
        #Generate plot
        s = io.BytesIO()
        plt.plot(csv_df_for_plot, linestyle='-', color="#5078b4")
        plt.xlabel('Zeit []')
        plt.ylabel('Wert []')
        plt.grid(True)
        plt.savefig(s, format='png',bbox_inches='tight')
        s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
        
        #Set max len to 1000, if len > 1000 entries
        newData_forpred_csv = pd.DataFrame(csv_df[:series_length])
        #bring data on len 1000 through repetition, if len < 1000 entries
        if len(newData_forpred_csv) < 1000:
            values = newData_forpred_csv.to_numpy()
            np_plus = np.tile(values,(int(series_length/values.shape[0]),1))
            rest = series_length - np_plus.shape[0]
            np_plus = np.concatenate((np_plus,values[:rest,]))
            newData_forpred_csv = pd.DataFrame(np_plus,columns = newData_forpred_csv.columns)
        
        
        newData_forpred_series_csv = newData_forpred_csv.squeeze(axis=1)      
        

        data_list_csv=[]
        newData_row_csv = [newData_forpred_series_csv]
        data_list_csv.append(newData_row_csv)

        formattest_csv = pd.DataFrame(data_list_csv, columns = ['X'])


        yhat_proba_csv = hc2rejclass.predict_proba(formattest_csv)
        yhat_csv = hc2rejclass.predict(formattest_csv)
        yhat_top3_type_csv = np.argsort(yhat_proba_csv, axis=1)[:,::-1][:,:3]
        #for cases where algorithm rejects device: prob value -> 0, type -> 9
        rej = (yhat_csv != 9)
        yhat_top3_type_csv[np.invert(rej)] = 9
            
        list_of_deviceclasses = ["vibration_sensor", "position_sensor",  "voltage_sensor", "power_sensor", 
                     "current_sensor", "velocity_sensor", "acceleration_sensor",
                     "temperature_sensor", "gyroscope", "no prediction"]

        proba_inner = yhat_top3_type_csv[0]

        predictions_for_FE = [list_of_deviceclasses[deviceclass] for deviceclass in proba_inner]
        
        with open('C:/Users/USER/.../foundData_objstores.pkl', 'wb') as knowndata_file_obj:
            pickle.dump(knownData_objstores, knowndata_file_obj)

        #Export data for node-red stream
        newData_found = {
            'newData': True,
            'database': 'Objektspeicher',
            'schema': path_fe,
            'table': files[0],
            'image': '<img src="data:image/png;base64,%s" alt="Plot">' % s,
            'prediction1': predictions_for_FE[0], 
            'prediction2': predictions_for_FE[1],
            'prediction3': predictions_for_FE[2]
            }
        newData_exp = json.dumps(newData_found)
        print(newData_exp)
        sys.exit()

if bool(newData) == False and bool(new_files) == False:
    newData_found = {
        'newData': False,
        }
    newData_exp = json.dumps(newData_found)
    print(newData_exp)
    sys.exit()

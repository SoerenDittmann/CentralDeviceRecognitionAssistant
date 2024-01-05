'''
Script to load required data into the working directory to execute the
simple example in the node-red flow.
'''
#%%Import packages
import io
import sys
import os
import pickle


#%%Read known Data
try:
    with open('C:/Users/USER/.../DataSimpleExample/knownData.pkl', 'rb') as knowndata_file:
        knownData_DB = pickle.load(knowndata_file)
except FileNotFoundError:
    print("Please update the path to the /DataSimpleExample directory")


try:
    with open('C:/Users/USER/.../DataSimpleExample/foundData_objstores.pkl', 'rb') as knowndata_file_obj:
        knownData_objstores = pickle.load(knowndata_file_obj)
except FileNotFoundError:
    print("Please update the path to the /DataSimpleExample directory")
    
#%%Write Data to directory used in the MonitoringDatabases_InclPlot_InclPred script
with open('C:/Users/USER/.../foundData_objstores.pkl', 'wb') as knowndata_file_obj:
    pickle.dump(knownData_objstores, knowndata_file_obj)

with open('C:/Users/USER/.../knownData.pkl', 'wb') as knowndata_file:
    pickle.dump(knownData_DB, knowndata_file)
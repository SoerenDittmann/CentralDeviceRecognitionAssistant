# CentralDeviceRecognitionAssistant
Node-RED based demonstrator of the central device recognition assistant approach from Semantische Assistenzen für komplexe Digitale Zwillinge (Dittmann, unv).

This repository contains the Node-RED flow of the implementation and the timeseries classifier in the Classifer directory.

To solely run the multiclass timeseries classifier, please consult the HowTo description in the Classifier directory.

Here a quick description regaring the Node-RED flow.

The following flow creates the dashboard for the central device recognition assistant. The frontend contains of four different parts:

1. A Banner notifying the user about newly found data in the virtual component
2. A prediction of the top three most likely classes of the newly found data
3. The semantics of the three most likely classes
4. Information regarding the newly found data: A plot and information about their location within the virtual component

The following code contains python files which require specific packages to run. For that only the following steps are required to run the code

1. Create an Anaconda Environment with the packages listed in env.txt
2. In the Anaconda Navigator, search for the environment and open a terminal from the environment
3. Type: node-red in the terminal to start a node-red instance within the created environment and the python scripts have access to the required packages
4. In the python-shell nodes that are part of the flow, no environment needs to be defined


Architecture:
- Apart from this flow, two additional components are required: A PostgreSQL database and the semantic description.
- For simplicity, the semantic description is added as a .json file within this repository. Please note that this file acts just as surrogate semantics.
- The PostgreSQL can be set up on the localhost or an external server. The flow currently expects the DB to be named as follows:

    Database: DigitalTwin_database
    Port: PostgreSQL default port 5432
    Schema: DigitalTwin_data_from_phys_comp
    Tables: ims_bearing

- Currently only the ims_bearing data is uploaded to the DB (see file 2003.11.01.18.01.44 in this repository)

  This data is provided by Lee, J., Qiu, H., Yu, G., Lin, J., Bearing data set, 2021
  Orginally av. https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
  Last acc. 14.02.2022
    

As mentioned this repository contains the env file containing the required conda packages to sucessfully run the python scripts.

The easiest way to install the packages is to either just read in the file in anaconda or firstly define the correct python version, 
then install the correct numpy, pandas, matplotlib, psycopg2 and sqlalchemy versions.

The Classifier directory contains the source code for the multiclass classification. This includes the initial search
for the model via a nested cross validation and the subsequent repeated cross validation with the chosen model incl. the final
metrics.


Overview of the files in this repository:
- env.txt: Packages to define the anaconda environment
- CentralDevRecAssistant_flow.json: Flow to import into the node-red instance started as discribed above. The path to this file needs to be configured in the "ÜberwachungDatenhaltung" node.
- MonitoringDatabases_InclPlot_InclPred.py: "Backend" of this demonstrator. Tracks the data base and object storage, detects and predicts new data and sends all information to the node-red flow.
	- Please update the paths in the python file to:
		- The connection details of the PostgreSQL database
		- The objectStore where your clone of https://github.com/SoerenDittmann/CentralDeviceRecognitionAssistant_objectStore is
		- The paths where you stored knownData.pkl and foundData_objstores.pkl is stored, if you would like to follow the simple example and just loaded data from DataSimpleExample within this repo
		- The path to the Final_Multiclass_Model.pkl from this repo
		- And the paths where the knownData.pkl and foundData_objstores.pkl should be stored

- information_models.json: Exemplary semantics. The path to this file needs to be configured in the "EinlesenSemantik" node in the node-red flow
- Final_Multiclass_Model: Pickle file containing the time series classificator.
- PrepareSimpleExample.py: Path to this file needs to be configured in the "LadenDerDaten" node, if simple example is wanted
	- Please set the paths to the files from the DataSimpleExample directory within this repo
- 2003.11.01.18.01.44: Exemplary data for the PostgreSQL as described above
- The Classification directory contains necessary helper functions for MonitoringDatabases_InclPlot_InclPred and should be stored in the same location. It was initially provided by Device recognition assistants as additional data management method for Digital Twins (Dittmann et al. unv)

The following processor was used to run the code in this repository: Intel(R) Core(TM) i7-8650U CPU @ 1.90GHz 2.11 GHz with 32 GB RAM.
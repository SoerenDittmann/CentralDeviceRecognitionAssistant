import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.io.sql as sqlio
import psycopg2
import sys

#Credit to Naysan Saran. https://naysan.ca/2020/06/21/pandas-to-postgresql-using-psycopg2-copy_from/
def connect(params_dic):
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params_dic)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        sys.exit(1) 
    print("Connection successful")
    return conn



def load_select_pad (connection, table_name):

    #---------Loading and selecting data----------------- 
    #Credit to Jake Brinkmann, https://gist.github.com/jakebrinkmann/de7fd185efe9a1f459946cf72def057e
    sql = "select * from "+str(table_name)+";" #select only certain columns select * from cwr_base4hp1730rpm;
    df = sqlio.read_sql_query(sql, connection)
    df = df.iloc[:,1:] #drop first (id) column because it cannot be omitted via SQL query directly 
    
    #-----------Normalize data column wise-----------------
    df = normalize(df)
    
    #-----------Pad to 50k entries-----------------
    series_length = 50000
    
    
    if df.shape[0] >= series_length:
        df_new = df.iloc[:series_length,:]
    else:
        values = df.to_numpy()
        np_plus = np.tile(values,(int(series_length/values.shape[0]),1))
        rest = series_length - np_plus.shape[0]
        np_plus = np.concatenate((np_plus,values[:rest,]))
        df_new = pd.DataFrame(np_plus,columns = df.columns)
    
    #print("Input Dataframe:")
    #print(df.shape, df.dtypes)
    #print("Padded Dataframe:")
    #print(df_new.shape, df_new.dtypes)

    #--------- return DataFrame----------------
    return df_new

def summary_plot (df):
    
    #---------Summarize and plot data--------------------
    plt.figure()
    print(df.describe())
    df.plot()
    plt.show()
    
def normalize(df):
    mean = np.mean(df,axis=0)
    std = np.std(df,axis=0)
    df = (df - mean[None,:])/std[None,:]
    return df
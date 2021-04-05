#run this line:
#pyenv version
#if not python 3.7.9
#pyenv install 3.7.9
import numpy as np
import pandas as pd
from IPython.display import display     #for display
from hmmlearn import hmm                #for the hidden markov model
import os 

#from hello import *             #imports entire file
#from hello import my_function  #imports sepcific function


def convertFiles(HR_list, Sleep_list):
    #HR_list = os.listdir(path='Data/HR')
    #Sleep_list = os.listdir(path='Data/Pre_Processed_Sleep')              #makes a list of all the files in folder

    for file in Sleep_list:
        with open("Data/Pre_Processed_Sleep/{}".format(file)) as infile, open("Data/Sleep/{}".format(file), 'w') as outfile:
            outfile.write(infile.read().replace(" ", ","))  #replaces spaces with ","
        




def Sleep_HR_data_processor(sleep_file, HR_file, fileNumber):        #files as .csv. fileNumber is the xth time function is called (used to name the files differently)

    df_sleep = pd.read_csv('Data/Sleep/{}'.format(sleep_file))
    df_sleep.columns =['TimeSec', 'SleepLVL']
    df_hr = pd.read_csv('Data/HR/{}'.format(HR_file))
    df_hr.drop_duplicates()
    df_hr.columns =['TimeSec', 'HR']

    #temp_df = df_sleep.iloc[:,0]                        #used to loop though the DF 
    df_Sleep_HR = df_sleep                              #coppies sleep time column and data column
    df_Sleep_HR["HR"] = 0                               #adds a column named HR

    for x in range(len(df_sleep)):
        #display(x)
        search_value = df_sleep.iloc[x, 0]                  #df.iloc[row,column]
        result_index = df_hr['TimeSec'].sub(search_value).abs().idxmin()    # returns row of hr data
        #display(df_hr.iloc[result_index, 1])           #displays the value found
        df_Sleep_HR.iloc[x, 2] = df_hr.iloc[result_index, 1]      #copy desired data to temp_df
    df_Sleep_HR.to_csv('Data/Merged/Sleep_HR_{}.csv'.format(fileNumber))                  #saves as a .csv 
    return df_Sleep_HR




def merge_Sleep_HR_Data(HR_list, Sleep_list):
    for fileNum in range(len(Sleep_list)):
        processed_data = Sleep_HR_data_processor(Sleep_list[fileNum], HR_list[fileNum], fileNum)      #iterates through all the files, merging sleep and HR data 

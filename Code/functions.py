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


def convertFiles(HR_list, Sleep_list):                      # Changes space to "," for Sleep files
    #HR_list = os.listdir(path='Data/HR')
    #Sleep_list = os.listdir(path='Data/Pre_Processed_Sleep')               # Makes a list of all the files in folder

    for file in Sleep_list:
        with open("Data/Pre_Processed_Sleep/{}".format(file)) as infile, open("Data/Sleep/{}".format(file), 'w') as outfile:
            outfile.write(infile.read().replace(" ", ","))  #replaces spaces with ","
        




def Sleep_HR_data_processor(sleep_file, HR_file, fileNumber):# files as .csv. fileNumber is the xth time function is called (used to name the files differently)

    df_sleep = pd.read_csv('Data/Sleep/{}'.format(sleep_file))
    df_sleep.columns =['TimeSec', 'SleepLVL']
    df_hr = pd.read_csv('Data/HR/{}'.format(HR_file))
    df_hr.drop_duplicates()
    df_hr.columns =['TimeSec', 'HR']

    df_Sleep_HR = df_sleep                                  #coppies sleep time column and data column
    df_Sleep_HR["HR"] = 0                                   #adds a column named HR

    for x in range(len(df_sleep)):
        #display(x)
        search_value = df_sleep.iloc[x, 0]                  #df.iloc[row,column]
        result_index = df_hr['TimeSec'].sub(search_value).abs().idxmin()    # returns row of hr data
        #display(df_hr.iloc[result_index, 1])                               #displays the value found
        df_Sleep_HR.iloc[x, 2] = df_hr.iloc[result_index, 1]                #copy desired data 
    df_Sleep_HR.to_csv('Data/Merged/Sleep_HR_{}.csv'.format(fileNumber))    #saves as a .csv 
    return df_Sleep_HR
 

 

def merge_Sleep_HR_Data(HR_list, Sleep_list):               # Links and adds HR data and Sleep data to one file
    for fileNum in range(len(Sleep_list)):
        processed_data = Sleep_HR_data_processor(Sleep_list[fileNum], HR_list[fileNum], fileNum)      #iterates through all the files, merging sleep and HR data 





def dataChecker():                                          # Checks for and deletes lines with SleepLVL values of -1
    MergedFileList = os.listdir(path='Data/Merged')
    for file in MergedFileList:                             # loop through files

        df = pd.read_csv('Data/Merged/{}'.format(file))     # opends csv file
        
        indexNames = df[ df['SleepLVL'] == -1 ].index       # Get names of indexes for which column SleepLVL has value -1
        df.drop(indexNames , inplace=True)                  # Delete these row indexes from dataFrame

        df = df.reset_index(drop=True)                      # resets the index numbers.
        df.columns =['index', 'TimeSec', 'SleepLVL', 'HR']
        df = df.drop(columns=['index'])
        df.to_csv('Data/Merged/{}'.format(file))            # saves as a .csv 

  

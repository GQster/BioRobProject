#run this line:
#pyenv version
#if not python 3.7.9
#pyenv install 3.7.9
import numpy as np
import pandas as pd
from IPython.display import display     #for display
from hmmlearn import hmm                #for the hidden markov model
import os 
#EXAMPLE
#from hello import *             #imports entire file
#from hello import my_function  #imports sepcific function


HR_list= ['1066528_heartrate.txt', '1360686_heartrate.txt', '1449548_heartrate.txt', '1455390_heartrate.txt', '1818471_heartrate.txt', 
                '2598705_heartrate.txt', '2638030_heartrate.txt', '3509524_heartrate.txt', '3997827_heartrate.txt', '4018081_heartrate.txt', 
                '4314139_heartrate.txt', '4426783_heartrate.txt', '46343_heartrate.txt', '5132496_heartrate.txt', '5383425_heartrate.txt', 
                '5498603_heartrate.txt', '5797046_heartrate.txt', '6220552_heartrate.txt', '759667_heartrate.txt', '7749105_heartrate.txt',
                '781756_heartrate.txt', '8000685_heartrate.txt', '8173033_heartrate.txt', '8258170_heartrate.txt', '844359_heartrate.txt', 
                '8530312_heartrate.txt', '8686948_heartrate.txt', '8692923_heartrate.txt', '9106476_heartrate.txt', '9618981_heartrate.txt', 
                '9961348_heartrate.txt']

Sleep_list= ['1066528_labeled_sleep.txt', '1360686_labeled_sleep.txt', '1449548_labeled_sleep.txt', '1455390_labeled_sleep.txt', '1818471_labeled_sleep.txt', 
                '2598705_labeled_sleep.txt', '2638030_labeled_sleep.txt', '3509524_labeled_sleep.txt', '3997827_labeled_sleep.txt', '4018081_labeled_sleep.txt', 
                '4314139_labeled_sleep.txt', '4426783_labeled_sleep.txt', '46343_labeled_sleep.txt', '5132496_labeled_sleep.txt', '5383425_labeled_sleep.txt', 
                '5498603_labeled_sleep.txt', '5797046_labeled_sleep.txt', '6220552_labeled_sleep.txt', '759667_labeled_sleep.txt', '7749105_labeled_sleep.txt',
                '781756_labeled_sleep.txt', '8000685_labeled_sleep.txt', '8173033_labeled_sleep.txt', '8258170_labeled_sleep.txt', '844359_labeled_sleep.txt', 
                '8530312_labeled_sleep.txt', '8686948_labeled_sleep.txt', '8692923_labeled_sleep.txt', '9106476_labeled_sleep.txt', '9618981_labeled_sleep.txt', 
                '9961348_labeled_sleep.txt']



def convertFiles(HR_list, Sleep_list):                      # Changes space to "," for Sleep files
    #HR_list = os.listdir(path='Data/HR')
    #Sleep_list = os.listdir(path='Data/Pre_Processed_Sleep')               # Makes a list of all the files in folder

    for file in Sleep_list:
        with open("Data/Pre_Processed_Sleep/{}".format(file)) as infile, open("Data/Sleep/{}".format(file), 'w') as outfile:
            outfile.write(infile.read().replace(" ", ","))  #replaces spaces with ","
            #end with
        #end for
    #end convertFiles


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
        #end for
    df_Sleep_HR.to_csv('Data/Merged/Sleep_HR_{}.csv'.format(fileNumber))    #saves as a .csv 
    return df_Sleep_HR
    #end Sleep_HR_data_processor

 
def merge_Sleep_HR_Data(HR_list, Sleep_list):               # Links and adds HR data and Sleep data to one file
    for fileNum in range(len(Sleep_list)):
        processed_data = Sleep_HR_data_processor(Sleep_list[fileNum], HR_list[fileNum], fileNum)      #iterates through all the files, merging sleep and HR data 
        #end for
    #end merge_Sleep_HR_Data


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
        #end for
    #end dataChecker

  
def Fixdata():                                               # Fixes all our data
    convertFiles(HR_list, Sleep_list)                      # Changes space to "," for Sleep files
    merge_Sleep_HR_Data(HR_list, Sleep_list)               # Links and adds HR data and Sleep data to one file

    # Sleep_HR files with negatives: 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 17, 19, 21, 22, 24, 25, 26, 27,28
    dataChecker()                                          # Checks for and deletes lines with SleepLVL values of -1
    #end Fixdata
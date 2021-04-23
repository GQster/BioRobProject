#run this line:
#pyenv version
#if not python 3.7.9
#pyenv install 3.7.9
import numpy as np
import pandas as pd
from IPython.display import display     #for display
from hmmlearn import hmm                #for the hidden markov model
import os 
import glob
import scipy
import sklearn
from sklearn import svm
from sklearn.metrics import accuracy_score  
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

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
Sleep_HR_list = ['Sleep_HR_0.csv', 'Sleep_HR_1.csv', 'Sleep_HR_2.csv', 'Sleep_HR_3.csv', 'Sleep_HR_4.csv', 'Sleep_HR_5.csv', 'Sleep_HR_6.csv', 'Sleep_HR_7.csv', 'Sleep_HR_8.csv', 'Sleep_HR_9.csv', 
                'Sleep_HR_10.csv', 'Sleep_HR_11.csv', 'Sleep_HR_12.csv', 'Sleep_HR_13.csv', 'Sleep_HR_14.csv', 'Sleep_HR_15.csv', 'Sleep_HR_16.csv', 'Sleep_HR_17.csv', 'Sleep_HR_18.csv', 'Sleep_HR_19.csv', 
                'Sleep_HR_20.csv', 'Sleep_HR_21.csv', 'Sleep_HR_22.csv', 'Sleep_HR_23.csv', 'Sleep_HR_24.csv', 'Sleep_HR_25.csv', 'Sleep_HR_26.csv', 'Sleep_HR_27.csv', 'Sleep_HR_28.csv', 'Sleep_HR_29.csv', 
                'Sleep_HR_30.csv']

def dataChecker():                                                          # Checks for and deletes lines with SleepLVL values of -1
    MergedFileList = os.listdir(path='HWData')
    print(MergedFileList)
    for file in MergedFileList:                                             # loop through files

        df = pd.read_csv('HWData/{}'.format(file))             		        # opends csv file
        
        indexNames = df[ df['SleepLVL'] == -1 ].index                       # Get names of indexes for which column SleepLVL has value -1
        df.drop(indexNames , inplace=True)                                  # Delete these row indexes from dataFrame

        df = df.reset_index(drop=True)                                      # resets the index numbers.
        df.columns =['index', 'TimeSec', 'SleepLVL', 'HR']
        df = df.drop(columns=['index'])
        df.to_csv('HWData/{}'.format(file))                            # saves as a .csv 
        #end for
    #end dataChecker


dataChecker()
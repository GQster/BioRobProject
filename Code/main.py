
#run this line:
#pyenv version
#if not python 3.7.9
#pyenv install 3.7.9
import numpy as np
import pandas as pd
from IPython.display import display     #for display
from hmmlearn import hmm                #for the hidden markov model

#from hello import *             #imports entire file
#from hello import my_function  #imports sepcific function




df_sleep = pd.read_csv("Data/1066528_labeled_sleep.csv")
df_sleep.columns =['TimeSec', 'SleepLVL']
df_hr = pd.read_csv("Data/1066528_heartrate.csv")
df_hr.drop_duplicates()
df_hr.columns =['TimeSec', 'HR']
display(df_sleep)
#display(df_hr)

temp_df = df_sleep.iloc[:,0]
#display(temp_df)
df_Sleep_HR = df_sleep                      #coppies sleep time column and data column

for x in temp_df:
    search_value = temp_df[x]               #df.loc[row,column]
   # print(search_value)
    result_index = df_hr['TimeSec'].sub(search_value).abs().idxmin()    # returns row of hr data
    df_Sleep_HR[x, 2] = df_hr[result_index, 1]  #copy desired data to temp_df
    display(result_index)
#display(result_index)
df_Sleep_HR.columns =['TimeSec', 'SleepLVL', 'HR']#Creates line labels 


#Hidden Mrkov Model
#model = hmm.GaussianHMM(n_components=3, covariance_type="full")

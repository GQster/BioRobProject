from functions import *     #imports our functions


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


#convertFiles(HR_list, Sleep_list)                      # Changes space to "," for Sleep files
#merge_Sleep_HR_Data(HR_list, Sleep_list)               # Links and adds HR data and Sleep data to one file

# Sleep_HR files with negatives: 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 17, 19, 21, 22, 24, 25, 26, 27,28
dataChecker()                                           # Checks for and deletes lines with SleepLVL values of -1




#Hidden Mrkov Model

# https://towardsdatascience.com/hidden-markov-model-implemented-from-scratch-72865bda430e
# https://jonathan-hui.medium.com/machine-learning-hidden-markov-model-hmm-31660d217a61


# df_merged = pd.read_csv('Data/Merged/{}'.format(sleep_file))
# df_merged.columns =['','TimeSec','SleepLVL','HR']
#model = hmm.GaussianHMM(n_components=6, covariance_type="full")
# n_components = 6                number of states in the model              (0, N1, N2, N3, N4, 5)
# X = 


import time
from random import random as rand
from pylsl import StreamInfo, StreamOutlet
import pandas as pd



def create_stream_from_csv(csv_df, type, sampling_freq, dtype='float32', stream_prefix='stream1'):
    header = list(csv_df.columns)
    n_channels = len(header)
    info = StreamInfo('CSV', type, n_channels, sampling_freq, dtype, stream_prefix)
    desc = info.desc()
    chns = desc.append_child('channels')
    for channel in header:
        chn = chns.append_child('channel')
        chn.append_child_value('label', channel)
        # Need unit functionality
    return StreamOutlet(info)


def get_data(fileName):
    if fileName == "combined_csv.csv":      #replaced get_dataALL
        df_file = pd.read_csv('/workspace/BioRobProject/Data/{}'.format(fileName))                # Creates a DF out of the csv file
        indexNames = ['Original Index', 'TimeSec']
        df_file.drop(indexNames , inplace=True, axis=1) 
        
    else:                   #replaces get_dataone
        df_file = pd.read_csv('/workspace/BioRobProject/Data/Merged/{}'.format(fileName))                # Creates a DF out of the csv file
        indexNames = ['Unnamed: 0', 'TimeSec']
        df_file.drop(indexNames , inplace=True, axis=1)
    return df_file


HRCSV = get_data('combined_csv.csv')


y = get_data('combined_csv.csv')

# split into input (X) and output (y) variables
del HRCSV['SleepLVL'] 
del y['HR'] 
#print(HRCSV)
#print(y)

#once per 30 seconds = 0.03333 Hz
fs = 15
CSVStream = create_stream_from_csv(HRCSV, type='Markers', sampling_freq=0.0666)           #not sue on the type

HRCSV = HRCSV.astype('float32')
print(HRCSV.dtypes)


print("now sending data...")
for idx, sample in HRCSV.iterrows():
    print(sample.values)
    CSVStream.push_sample(sample.values)
    time.sleep(fs)

print('End of CSV Reached, Closing Stream...')



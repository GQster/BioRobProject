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


directory = 'Data/'
EMGCSV = pd.read_csv(directory + 'Myo/emgData/RawEMG/Bangli/EMG011.txt', sep= ' ',
                      names=['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6', 'EMG7','EMG8',
                             'EMG9', 'EMG10', 'EMG11', 'EMG12', 'EMG13', 'EMG14', 'EMG15','EMG16',
                             'Class', 'Unknown'])

fs = 1000
CSVStream = create_stream_from_csv(EMGCSV, type='EEG', sampling_freq=512)

EMGCSV = EMGCSV.astype('float32')
print(EMGCSV.dtypes)

print("now sending data...")
for idx, sample in EMGCSV.iterrows():
    #print(sample.values)
    if (sample.values[5] < -213748364) or (sample.values[5] > 2147483647):
        print('Here')
    CSVStream.push_sample(sample.values)
    time.sleep(1/fs)

print('End of CSV Reached, Closing Stream...')
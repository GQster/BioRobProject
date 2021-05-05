import pylsl
import asyncio
import logging
#from keras.utils import to_categorical
from Util import util, ml
#import Util.util
#import Util.ml
import pandas as pd
from pandas import Timestamp
from datetime import datetime, timedelta
import os
from configparser import ConfigParser

import numpy as np
import tensorflow




logger = logging.getLogger(__name__)


async def save_current_streams(directory, dataPath, file_name_prefix):
    stream_names = []
    print("looking for streams")
    streams = pylsl.resolve_streams()
    for stream in streams:
        stream_names.append(stream.name())
        print(stream.name())
        print(util.obtain_stream_channel_names(stream))

    # Initialize Inlets and Dataframes
    inlets = []
    df_dict = {}
    path = os.path.join(directory, dataPath)
    for stream in streams:
        inlets.append(pylsl.StreamInlet(stream))
        header = util.obtain_stream_channel_names(stream)
        header.append('Device_Time')
        df_dict[stream.name()] = pd.DataFrame(columns=header)
        # Create directory for files
        try:
            os.makedirs(os.path.join(path, stream.name()))
        except:
            logger.info('Directory ', os.path.join(path, stream.name()), 'already exists')


    # Initialize Model and Pipeline
    clf = ml.emg_clf()
    #clf.load_model(modelInfo['directory'] + modelInfo['model'])
    clf.load_model('Saved_Models/all_cnn')
    pipeline = ml.ml_pipeline()
    #pipeline.add(ml.filter_all_channels)
    pipeline.add(clf.format_cnn_data)
    pipeline.add(clf.model.predict)
    pipeline.add(np.argmax)

    #window_size = int(1)
    stride = int(3)
    input_x = int(2)

    inlet = inlets[0]
    inlet_name = inlet.info().name()
    file_name = os.path.join(path, inlet_name, file_name_prefix + '_data.csv')
    pred_name = os.path.join(path, inlet_name, file_name_prefix + '_pred.csv')

    stream_df = pd.DataFrame(columns=header)
    next_update = Timestamp(0).now() + timedelta(seconds=stride)
    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        samples, timestamps = inlet.pull_chunk()                            #this line pulls from the stream
        if timestamps:

            # Save samples to csv
            sample_df = df_dict[inlet_name]
            df_temp = util.format_data_into_dataframe(samples, timestamps, sample_df.columns.values.tolist())
            sample_df = sample_df.append(df_temp)
            df_dict[inlet_name] = sample_df
            hdr = False if os.path.isfile(file_name) else True
            sample_df.to_csv(file_name, mode='a', index_label='Timestamp', header=hdr)
            col_names = [i for i in sample_df.columns]
            df_dict[inlet_name] = pd.DataFrame(columns=col_names)

            # Append samples to the stream df
            stream_df = stream_df.append(sample_df)

            # See if we have enough samples to make a predicition on
            if len(stream_df) < input_x:
                continue
            # Save predictions every update
            if Timestamp(0).now() >= next_update:
                # get the last input_x samples
                #windowed_df = stream_df.iloc[-input_x:]
                windowed_df = stream_df.iloc[-1:]
                # update stream_df to manage space
                stream_df = windowed_df
                # predict using our pipeline
                prediction = pipeline.predict(windowed_df)
                # determine the true class from the windowed_df
                target = max(windowed_df['HR'].values)
                # get the current time
                time = Timestamp(0).now()
                # print everything
                print("Time                            Prediction   HR data point")
                print(time, "    ",prediction, "          ",target)

                # Initialize a new dataframe and add our data to it
                pred_df = pd.DataFrame()
                pred_df['Timestamp'] = [time]
                pred_df['Prediction'] = [prediction]
                pred_df['Target'] = [target]

                # Append everything to a csv file
                pred_hdr = False if os.path.isfile(pred_name) else True
                pred_df.to_csv(pred_name, mode='a', index_label='Timestamp', header=pred_hdr)
                next_update = Timestamp(0).now() + timedelta(seconds=stride)








async def main():
    #directory, dataPath, file_name_prefix = get_file_info_from_config()
    directory = "/workspace/BioRobProject/Data/"
    dataPath = "Sleep_HR_0.csv"
    #dataPath = "combined_csv.csv"
    file_name_prefix = "Live_Stream"
    await save_current_streams(directory, dataPath, file_name_prefix)



if __name__ == '__main__':
    asyncio.ensure_future(main())
    loop = asyncio.get_event_loop()
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        logger.info("Ctrl-C pressed.")
    finally:
        loop.close()
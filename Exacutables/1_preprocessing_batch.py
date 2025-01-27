import sys, os, glob
import Enum.enum_types as et
from scipy.io import wavfile
from Feature_Extraction import feature_extractor as fe
from Outputs import data_writter as dw
from datetime import date,time, datetime, timedelta
from pandas import date_range
from pandas import DatetimeIndex
from datetime import datetime, timedelta
from Filtering import filtering as filt
import joblib
import numpy as np

def parameter_check_and_handling(argv:list):
    if len(argv) <= 11:
        print("Expected parameters: \n "
              "1 - event name\n"
              "2 - hive id \n"
              "3 - event date in format yyyy.mm.dd\n"
              "4 - event time in format hh:mm:ss\n"
              "5 - data before event in weeks(w)\n"
              "6 - number of features (DIM 1), \n"
              "7 - hop_length, \n"
              "8 - feature type to extract \n"
              "9 - filter type to apply\n"
              "10 - lower cutoff frequency\n"
              "11 - higher cutoff frequency\n"
              f"number of received parameters: {len(argv)}")
        sys.exit(1)

    if argv[1] not in et.Event_type._value2member_map_:
        print(f"{argv[1]} not recognised as a valid event type. ERROR")
        sys.exit(1)
    else:
        event_type = et.Event_type(argv[1])

    try:
        n_hive_id = int(argv[2])
    except ValueError:
        print(f"unable to cast 2nd parameter: {argv[2]} to integer as hive id. ERROR")
        exit(1)

    try:
        data = str(argv[3]).split('.')
        d_event_date = date(int(data[0]),int(data[1]),int(data[2]))
    except ValueError:
        print(f"unable to cast 3rd parameter: {argv[3]} to date as event date. ERROR")
        exit(1)

    try:
        data = str(argv[4]).split(':')
        t_event_time = time(int(data[0]),int(data[1]),int(data[2]))
    except ValueError:
        print(f"unable to cast 4th parameter: {argv[4]} to time as event time. ERROR")
        exit(1)

    try:
        n_range_in_weeks = int(argv[5])
    except ValueError:
        print(f"unable to cast 5th parameter: {argv[5]} to integer as range in weeks. ERROR")
        exit(1)

    try:
        n_feature = int(argv[6])
    except ValueError:
        print(f"unable to cast 6th parameter: {argv[6]} to integer as number of features. ERROR")
        exit(1)

    try:
        n_hop_length = int(argv[7])
    except ValueError:
        print(f"unable to cast 7th parameter: {argv[7]} to integer as hop length. ERROR")
        exit(1)

    if argv[8] not in et.Feature_type._value2member_map_:
        print(f"{argv[8]} not recognised as a valid feature type. ERROR")
        sys.exit(1)
    else:
        if argv[8] == 'all':
            feature_types = et.Feature_type.list()
            feature_types.pop()     # remove last element "all" from list
        else:
            feature_types = [et.Feature_type(argv[8])]

    if argv[9] not in et.Filtering_Type._value2member_map_:
        print(f"{argv[9]} not recognised as a valid filter type. ERROR")
        sys.exit(1)
    else:
        if argv[9] == 'all':
            filter_types = et.Filtering_Type.list()
            filter_types.pop()      # remove last element "all" from list
            filter_types.pop()      # remove last element "none" from list
            #filter_types.pop()      # remove last element "dani" from list
        else:
            filter_types = [et.Filtering_Type(argv[9])]

    try:
        param1 = int(argv[10])
    except ValueError:
        print(f"unable to cast 6th parameter: {argv[10]} to integer as param1 cut of frequency 1. ERROR")
        exit(1)

    try:
        param2 = int(argv[11])
    except ValueError:
        print(f"unable to cast 7tp parameter: {argv[11]} to integer as param2 cut of frequency 2. ERROR")
        exit(1)

    cutoff = (param1,param2)

    return event_type, n_hive_id, d_event_date, t_event_time, \
           n_range_in_weeks, n_feature, n_hop_length, feature_types, filter_types, cutoff

def check_directories(parent_dir:str, source_dir:str, target_dir:str):
    if not os.path.isdir(parent_dir+source_dir):
        print(f'Missing source directory: {parent_dir+source_dir}')
        sys.exit(1)

    if not os.path.isdir(parent_dir+target_dir):
        os.mkdir(parent_dir+target_dir)
        print(f'Missing target directory. Creating: {target_dir}')

def create_filename_list(hive_id:int, event_date:date, event_time:time, n_range_in_weeks:int):

    return_file_names = []
    n_range_id_days = n_range_in_weeks * 7
    event_date_time = datetime.combine(event_date, event_time)

    # get all files, belonging to the specific hive id
    hive_id_files = glob.glob(f'*-{hive_id}.wav')


    for file_part in hive_id_files:
        date_time = file_part.replace(f'-{hive_id}.wav', '')
        date_time = datetime.strptime(date_time, '%y%m%d-%H%M%S')
        delta_in_days = (event_date_time - date_time).days

        # if file's date is in range append to return list
        if -1 <= delta_in_days <= n_range_id_days:
            return_file_names.append(file_part)

    # sort and then return list
    return_file_names.sort()
    return return_file_names

def split_filename(filename:str):
    temp_record_date, temp_record_time, temp_hive_id = filename.split('-')
    hive_id = temp_hive_id.split('.')[0]
    record_date = datetime.strptime(temp_record_date, '%y%m%d').strftime("%Y_%m_%d")
    record_time = datetime.strptime(temp_record_time, '%H%M%S').strftime("%H:%M:%S")

    #hive_id = "0"
    #record_date = "2000_01_01"
    #record_time = "12:01:59"
    return hive_id, record_date, record_time

if __name__ == "__main__":

    # --------------------|
    # RUNNING PARAMETERS: |
    # --------------------|
    # swarming 24 2020.06.01 10:00:00 2 128 128 all all 1 3999
    # evet_name, hive_id, event_date, event_time,
    # weeks before, num of features, hop_length, feature type,
    # filter_type, lower cutoff, higher cutoff

    # nyq freq = 0.5 * sampling rate
    # Cutoff = (cutoff_freq[0]/nyq, cutoff_freq[1]/nyq)
    # all values of cutoff must be 0 < ... < 1
    # selected cutoffs: 1 and 3999 Hz

    # extracts all specified features belonging to a specific hive_id, when an event occurs in a specific date
    # and time. Filtering out all files within the range (in weeks) of event timedate..

    # parameter check and handling. Setting values from ENUM...
    e_event_type, n_hive_id, d_event_date, \
    t_event_time, n_range_in_weeks, n_features, \
    n_hop_length, feature_types, filter_types, cutoff = parameter_check_and_handling(sys.argv)

    source_dir = "/DATA/WAV"
    target_dir = "/DATA/FE"
    #parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    # check if directories are existing, if not ERROR
    check_directories(parent_dir,source_dir, target_dir)


    # change working directory to source directory
    os.chdir(parent_dir+source_dir)

    # create list of files belonging to a specific hive in alphabetical order
    l_list_of_files_in_order = create_filename_list(n_hive_id,d_event_date,t_event_time,n_range_in_weeks)

    mapping=[]
    for feat_type in feature_types:
        for filt_type in filter_types:
            mapping.append(f"{feat_type.value}-{filt_type.value}")

    with open(parent_dir+"/DATA/feature_list.joblib", 'wb') as f:
        joblib.dump(mapping, f)

    mapping2={}

    try:
        for s_file_name in l_list_of_files_in_order:
            hive_id, record_date, record_time = split_filename(s_file_name)
            n_sampling_rate,wav_data = wavfile.read(f'{s_file_name}')
            wav_data = wav_data.astype(float)

            for feat_type in feature_types:

                feature_set = fe.Fit(data=wav_data, type=feat_type, hop_length=n_hop_length,
                                     n_features=n_features, sampling_rate=n_sampling_rate, n_fft=10)

                for filt_type in filter_types:

                    filt_feature_set = filt.Fit(data=feature_set, type=filt_type, sampling_rate=n_sampling_rate,
                                            separation_logic=et.separation_type.BY_FREQ_SLICES.value,
                                            cutoff=cutoff, order=1)
                    print(f'{feat_type.value} - {filt_type.value} - Data dimensions: ', filt_feature_set.shape)
                    mapping2[f"{feat_type.value}-{filt_type.value}"]=filt_feature_set.shape

                    #dw.write_data_to_h5(directory=parent_dir + target_dir,
                    #                  filename= f"{hive_id}_{record_date}_{record_time.replace(':','_')}_{feat_type.value}_{filt_type.value}",
                    #                  ds_label= feat_type.value,data=filt_feature_set)

            start = datetime(int(record_date[0:4]),int(record_date[5:7]),int(record_date[8:]),
                         int(record_time[0:2]),int(record_time[3:5]),int(record_time[6:]))

            end = start + timedelta(seconds=(len(wav_data)/n_sampling_rate))
            timestamps_pydate = date_range(start,end,feature_set.shape[1]).to_pydatetime()
            timestamps_str = []

            for i in timestamps_pydate:
                timestamps_str.append(i.strftime("%Y-%m-%d %H-%M-%S-%f"))

            dw.write_data_to_h5(directory=parent_dir + target_dir,
               filename= f"{hive_id}_{record_date}_{record_time.replace(':','_')}_timestamps",
               ds_label= "timelabels",data=timestamps_str)
        with open(parent_dir+"/DATA/feature_list_with_dim.joblib", 'wb') as f:
            joblib.dump(mapping2, f)

    except Exception as inst:
        print(f"==============  ERROR: {s_file_name}  ==============")
        print(inst)  # __str__ allows args to be printed directly,
        print(f"=============================================")

    finally:

        del cutoff, d_event_date, feat_type, feature_set, feature_types, filt_feature_set, filt_type,\
            filter_types, hive_id, l_list_of_files_in_order, n_features, n_hive_id, n_hop_length,\
            n_range_in_weeks, n_sampling_rate, parent_dir, record_date, record_time, s_file_name, \
            source_dir, t_event_time, target_dir, e_event_type, timestamps_pydate ,timestamps_str, wav_data

    sys.exit(0)

#TODO: Need mapping of Features

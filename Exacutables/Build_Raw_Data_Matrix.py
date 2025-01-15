import os, sys, glob
import Enum.enum_types as et
from datetime import date, time, datetime
import h5py
import numpy as np
from Outputs import data_writter as dw


def parameter_check_and_handling(argv:list):
    if len(argv) <= 5:
        print("Expected parameters: \n "
              "1 - event name\n"
              "2 - hive id list in format [1,2,3]\n"
              "3 - event date in format yyyy.mm.dd\n"
              "4 - event time in format hh:mm:ss\n"
              "5 - data before event in weeks(w)\n"
              f"number of received parameters: {len(argv)}")
        sys.exit(1)

    if argv[1] not in et.Event_type._value2member_map_:
        print(f"{argv[1]} not recognised as a valid event type. ERROR")
        sys.exit(1)
    else:
        event_type = et.Event_type(argv[1])

    try:
        l_hive_id = list(map(int, sys.argv[2].strip('[]').split(',')))
    except ValueError:
        print(f"unable to cast 2nd parameter: {argv[2]} to a list of integers as hive id list. ERROR")
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

    return event_type, l_hive_id, d_event_date, t_event_time, n_range_in_weeks

# list with all features
def create_feature_list():
    list = et.Feature_type.list()
    list.remove(et.Feature_type.ALL)
    return list


def create_filename_list(hive_id:int, event_date:date, event_time:time,
                         n_range_in_weeks:int, feature_name:str):

    return_file_names = []
    n_range_id_days = n_range_in_weeks * 7
    event_date_time = datetime.combine(event_date, event_time)

    hive_id_files = glob.glob(f'{hive_id}*{feature_name}.h5')

    for file in hive_id_files:

        date_time = file[3:].replace(f'_{feature_name}.h5','')
        #date_time = datetime.strptime(date_time, '%y%m%d-%H%M%S')
        date_time = datetime.strptime(date_time,'%Y_%m_%d_%H_%M_%S')
        delta = (event_date_time - date_time).days

        if -1 <= delta <= n_range_id_days:
            return_file_names.append(file)
    return_file_names.sort()
    return return_file_names


def create_feature_data(dir_path:str,hi:int, fi:int, feature: str, raw_fl_3D:np.ndarray ):
    fr_data = []
    j = 0
    for t in range(raw_fl_3D.shape[2]):  # [2020_05_24, 2020_05_25, 2020_05_26]
        f_name = raw_fl_3D[hi][fi][t]
        h5file_data = h5py.File(dir_path + "/" + f_name, 'r')
        if j == 0:
            fr_data = np.asarray(h5file_data.get(feature))
            j = 1
        else:
            fr_data = np.append(fr_data, np.asarray(h5file_data.get(feature)), axis=1)
    return fr_data


def create_hive_data_matrix(dir_path:str, hive_idx:int, feature_list:list, raw_fl_3D:np.ndarray ): # for all features
    data = []
    i=0
    for fi in range(raw_fl_3D.shape[1]):     # [stft, mfcc]
        if i==0:
            data = create_feature_data(dir_path, hive_idx, fi, feature_list[fi].value , raw_fl_3D)
            i=1
        else:
            data = np.append(data,create_feature_data(dir_path, hive_idx, fi, feature_list[fi].value , raw_fl_3D),axis = 0 )
    return data


# >
if __name__ == "__main__":

    # swarming [24] 2020.06.01 10:00:00 2
    # event name, hive id list, event date, event time, weeks before event

    source_dir = "/Feature_Identification/DATA/FE"
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    dir_path = parent_dir + source_dir

    os.chdir(dir_path)

    # parameter check
    event_type, l_hive_id, d_event_date, t_event_time, n_range_in_weeks = parameter_check_and_handling(sys.argv)

    # get list of all possible features
    feature_list = create_feature_list()

    # build up file structure matrix
    j=0
    for hive_id in l_hive_id:           # extend list in 1st dimension
        i=0
        for feature in feature_list:
            _list = [create_filename_list(hive_id, d_event_date, t_event_time,n_range_in_weeks, feature.value)]
            if i==0:
                raw_fl_2D = np.asarray(_list)
                i=i+1
            else:
                raw_fl_2D = np.insert(raw_fl_2D,len(raw_fl_2D),np.asarray(_list),axis=0)
        if j==0:
            raw_fl_3D = np.asarray([raw_fl_2D])
            j=j+1
        else:
            raw_fl_3D = np.insert(raw_fl_3D, len(raw_fl_3D), np.asarray(raw_fl_2D), axis=0)

    del raw_fl_2D, j, i, _list, hive_id
    # raw_fl_3D - contains filenames in the format in which data will be needed


    # build up 3d raw data matrix
    raw_data_matrix=[]
    h = 0
    for h_idx in range(raw_fl_3D.shape[0]):   # [24,26]
        if h==0:
            raw_data_matrix = create_hive_data_matrix(dir_path, h_idx, feature_list, raw_fl_3D)
            raw_data_matrix = np.expand_dims(raw_data_matrix, axis=0)
            h=1
        else:
            raw_data_matrix = np.insert(raw_data_matrix, raw_data_matrix.shape[0],
                                        create_hive_data_matrix(dir_path, h_idx, feature_list, raw_fl_3D), axis=0)

    del h, feature_list

    # Write out raw data matrix
    dw.write_data_to_h5(directory=parent_dir + source_dir, filename=f"raw_data_matrix_{'_'.join(map(str,l_hive_id))}",
                        ds_label="all", data=raw_data_matrix)

    # (2, 256, 1539)
    # time labels: read, concatenate, write out

    # concatenate timestamps and write them out
    del raw_data_matrix, h_idx
    k = 0
    for i in range(raw_fl_3D.shape[0]):
        for j in raw_fl_3D[i][0]:
            # read file(s) and concatante
            h5file_data = h5py.File(dir_path + "/" + j[:-3]+"_timestamps.h5", 'r')
            if k == 0:
                label_data = np.asarray(h5file_data.get("timelabels"))
                k = 1
            else:
                label_data = np.append(label_data, np.asarray(h5file_data.get("timelabels")), axis=0)

        dw.write_data_to_h5(directory=parent_dir + source_dir, filename=f"raw_data_timestamps_{j[:2]}",
                            ds_label="all", data=label_data)

    del raw_fl_3D, label_data, d_event_date,dir_path,event_type, feature, h5file_data, i,j,k, l_hive_id,\
        n_range_in_weeks, parent_dir, source_dir, t_event_time

    print("Done")

import os, json



'''
 -1: invalid
 0: static or walking
 1: MTR
'''
transportation_tag = [1, 2, 3, 4, 5]
valid_tag = [0, 1, 2, 3, 4, 5]

logname = 'MajorLog.log'

# These are all the features recorded by our devices in every piece of data. The data on the server has one piece per minute.
# You may select the features, if necessary, which are all included in this list, that you wanna fetch and modify the variable {feature_list} at the bottom part of this script correspondingly.
feature_list = ['NodeID', 'SubSeqNo', 'Time', 'Date', 'GPSTime', 'GPSDate', 
            'NO2_WE_uV', 'NO2_AE_uV', 'NO_WE_uV', 'NO_AE_uV', 'CO_WE_uV', 'CO_AE_uV', 'O3_WE_uV', 'O3_AE_uV', 
            'T_C', 'RH_PER', 'Tadj', 'RH_adj', 
            'Mic_mean', 'Mic_min', 'Mic_max', 'Mic_SD', 
            'Cnts_GT_Th1_MIC', 'Dur_GT_Th1_MIC', 'Cnts_GT_Th2_MIC', 'Dur_GT_Th2_MIC', 
            'Cnts_LT_Th3_MIC', 'Dur_LT_Th3_MIC', 'Cnts_LT_Th4_MIC', 'Dur_LT_Th4_MIC', 
            'SVM_mean', 'SVM_min', 'SVM_max', 'SVM_SD', 
            'Cnts_GT_Th1_SVM', 'Dur_GT_Th1_SVM', 'Cnts_GT_Th2_SVM', 'Dur_GT_Th2_SVM', 
            'Cnts_LT_Th3_SVM', 'Dur_LT_Th3_SVM', 'Cnts_LT_Th4_SVM', 'Dur_LT_Th4_SVM', 
            'Batt_V', 'Input_V', 'Lon', 'Lat', 'Alt', 'Sats', 'HDOP', 'Fix_qual', 
            'PM1', 'PM25', 'PM10', 'SFR', 'Period_Cnt', 
            'Bin_0', 'Bin_1', 'Bin_2', 'Bin_3', 'Bin_4', 'Bin_5', 'Bin_6', 'Bin_7', 
            'Bin_8', 'Bin_9', 'Bin_10', 'Bin_11', 'Bin_12', 'Bin_13', 'Bin_14', 'Bin_15', 
            'Bin_16', 'Bin_17', 'Bin_18', 'Bin_19', 'Bin_20', 'Bin_21', 'Bin_22', 'Bin_23', 
            'OPC_Temp', 'OPC_RH', 'OPC_RCNT_GLCH', 'OPC_RCNT_LTOF', 'OPC_RCNT_RAT', 'OPC_RCNT_OORNG', 
            'OPC_FAN_CNT', 'OPC_LSR_STAT', 'OPC_MTOF0', 'OPC_MTOF1', 'OPC_MTOF2', 'OPC_MTOF3', 'RadioCNT']

# device ID. 305 for Owen, 306 for Johnny and 312 for Saunders.
nodes_list = ["305", "306", "312"]  


def loadData(serial, date) -> list:
    filename = 'data\\' + serial + '-' + date + '.json'
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f'\nData from device #{serial} on {date} successfully loaded.\n{len(data)} lines of data loaded in total.\nFormat: [Time, Date, SVM_mean, SVM_min, SVM_max, SVM_SD]\n')
    except Exception as e:
        print(f'\nFailed. Error: {e}\n')
    return data

def loadLabelledData(serial, date) -> list:
    filename = 'data\\labelled-' + serial + '-' + date + '.json'
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f'\nLabelled data from device #{serial} on {date} successfully loaded.\n{len(data)} lines of data loaded in total.\nFormat: [Time, Date, SVM_mean, SVM_min, SVM_max, SVM_SD]\n')
    except Exception as e:
        print(f'\nFailed. Error: {e}\n')
    return data

def loadCaliData(serial, date) -> list:
    filename = 'data\\cali-' + serial + '-' + date + '.json'
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f'\nCalibrated data from device #{serial} on {date} successfully loaded.\n{len(data)} lines of data loaded in total.\nFormat: [Time, Date, SVM_mean, SVM_min, SVM_max, SVM_SD]\n')
    except Exception as e:
        print(f'\nFailed. Error: {e}\n')
    return data

def save_labelled_data(d, date, serial) -> None:
    filename = f'data\\labelled-{serial}-{date}.json'
    try:
        with open(filename, 'w+') as f:
            json.dump(d, f)
        log_write(f'\n{len(d)} numbers of labelled data successfully saved to {filename}\n')
    except Exception as e:
        log_write(f'\nFailed to store labelled data. Error: {e}\n')
    return

def save_cali_data(d, date, serial) -> None:
    filename = f'data\\cali-{serial}-{date}.json'
    try:
        with open(filename, 'w+') as f:
            json.dump(d, f)
        log_write(f'\n{len(d)} numbers of calibrated data successfully saved to {filename}\n')
    except Exception as e:
        log_write(f'\nFailed to store calibrated data. Error: {e}\n')
    return

def time_match(running, instant) -> bool:
    if type(instant) == str:
        instant = int(instant[0:4])

    for (start, end, mode) in running:
        if start - 800 <= instant and end - 800 >= instant:
            return mode
        else:
            continue
    return 0

def log_write(s) -> None:
    global logname
    with open(logname, 'a') as f:
        print(s, file=f)
        print(s)
    return
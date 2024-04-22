import os, json, random, time



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

# MTR = 1, Bus = 2, LRT = 3
d312_20240101 = [(1418, 1427, 1), (1555, 1605, 1), (1852, 1858, 1), (1903, 1907, 1), (1914, 1949, 1), (1957, 2032, 1), (2037, 2041, 1), (2047, 2053, 1), (2133, 2201, 1), (2206, 2233, 1)]
d312_20240102 = [(1916, 1924, 1), (1925, 1934, 1), (1953, 2004, 1), (2010, 2039, 1), (2042, 2121, 1), (2128, 2132, 1), (2149, 2230, 1), (2237, 2323, 1), (2328, 2333, 1)]
d312_20240103 = [(1436, 1441, 1), (1447, 1457, 1), (1459, 1509, 1), (1514, 1554, 1), (1600, 1640, 1), (1646, 1651, 1), (1911, 1916, 1), (1921, 1926, 1), (1928, 2005, 1), (2010, 2045, 1), (2047, 2052, 1), (2058, 2063, 1)]
d312_20240106 = [(1539, 1543, 1), (1546, 1552, 1), (1559, 1636, 1), (1640, 1708, 1), (1736, 1758, 1), (1814, 1858, 2)]
d312_20240107 = [(1500, 1504, 1), (1507, 1526, 1), (1537, 1554, 1), (1611, 1657, 3), (1701, 1748, 3), (1825, 1837, 1), (1844, 1854, 1), (1858, 1920, 1), (1922, 1925, 1), (1928, 1957, 1), (1958, 2001, 1), (2007, 2023, 1)]
d312_20240119 = [(1941, 1948, 1), (1951, 2044, 1)]
# d312_20240127 = [(1604, 1610, 1), (1616, 1656, 1), (2230, 2310, 1), (2315, 2323, 1)]
d312_20240329 = [(1445, 1451, 1), (1457, 1542, 1)]
d312_20240415 = [(1957, 2018, 2), (2034, 2110, 2)]
d312_20240417 = [(1607, 1620, 2), (1631, 1652, 2)]
d312_20240419 = [(1647, 1702, 2), (1713, 1803, 2), (1823, 1906, 2)]

dates = ['20240101', '20240102', '20240103', '20240106', '20240107', '20240119', '20240329', '20240415', '20240417', '20240419']

def storeData(data, serial, date) -> None:
    try:
        os.makedirs('data')
    except:
        pass  # folder already exists

    filename = 'data\\' + serial + '-' + date + '.json'
    try:
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f'\nData on {date} successfully fetched.\nStored at {filename}\n')
    except Exception as e:
        print(f'\nFailed to store data. Error: {e}\n')
    return

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
        print(f'\nCalibrated data from device #{serial} on {date} successfully loaded.\n{len(data)} lines of data loaded in total.\nFormat: [Time, Date, SVM_mean, SVM_min, SVM_max, SVM_SD, Tag]\n')
        cnt = 0
        for line in data:
            if line[-1] > 0:
                cnt += 1
        print(f'{cnt} numbers of dynamic data loaded.')
    except Exception as e:
        print(f'\nFailed. Error: {e}\n')
    return data

def load_seg_data(seg_length, offset, flag=None) -> list:
    if flag == "confi":
        filename = f'data\\segsConfi_{seg_length}_{offset}.json'
    elif flag == None:
        filename = f'data\\segs_{seg_length}_{offset}.json'
    else:
        print(f'Wrong flag: {flag} in function {load_seg_data.__name__}')
        exit(1)

    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f'\nSegmentations on {filename} successfully loaded, {len(data)} numbers of segs in total.\nFormat: [[SVM_mean, SVM_min, SVM_max, SVM_SD, Tag], ...]\n')
        cnt = 0
        for seg in data:
            if seg[0][-1] > 0:
                cnt += 1
        print(f'{cnt} numbers of dynamic data included.')
    except Exception as e:
        print(f'\nFailed. Error: {e}\n')
    return data

def load_train_data(seg_length, offset, testing_set_length, mode) -> None:
    
    data = load_seg_data(seg_length, offset)
    training_set = []
    testing_set = []
    cnt = 0
    random.seed(time.time())
    random.shuffle(data)
    # for seg in data:
    #     if seg[-1] == mode:
    #         print(seg)

    for seg in data:
        if len(testing_set) < testing_set_length: 
            if cnt < 5:
                if seg[-1] != mode:
                    training_set.append(seg)
                else:
                    testing_set.append(seg)
                    cnt += 1
            else:
                testing_set.append(seg)
        else:
            training_set.append(seg)
    print(f'training set length: {len(training_set)}  testing set length: {len(testing_set)}')
    return training_set, testing_set
    
def LoadCatagory(serial, dates) -> None:
    dataSet = Data_Set()
    for date in dates:
        data = loadLabelledData(serial, date)
        for line in data:
            dataSet.add(Data_Unit(line))
    return dataSet

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

def save_seg_data(d, filename, flag=None) -> None:
    try:
        with open(filename, 'r+') as f:
            data = json.load(f)
    except:
        data = []
    try:
        for line in d:
            data.append(line)
        with open(filename, 'w+') as f:
            json.dump(data, f)
        log_write(f'\n{len(d)} numbers of segmentation data successfully appended to {filename}\n')
    except Exception as e:
        log_write(f'\nFailed to store segmentation data. Error: {e}\n')
    return

def save_confi_data(data, seg_length, offset) -> None:
    filename = f'data\\confi_{seg_length}_{offset}.json'
    try:
        with open(filename, 'a+') as f:
            json.dump(data, f)
        log_write(f'\n{len(data)} numbers of segmentation data with labelled confidence successfully saved to {filename}\n')
    except Exception as e:
        log_write(f'\nFailed to store confidence data. Error: {e}\n')
    return

def save_train_data(trainSet, testSet, seg_length, offset) -> None:
    try:
        with open(f'training\\train_{seg_length}_{offset}.json', 'w+') as f:
            json.dump(trainSet, f)
        with open(f'training\\test_{seg_length}_{offset}.json', 'w+') as f:
            json.dump(testSet, f)
        log_write(f'\nTraining set and testing set ready.\n')
    except Exception as e:
        log_write(f'\nFailed on function {save_train_data.__name__}. Error: {e}\n')
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

def log_write(s, path = None) -> None:
    global logname
    if path != None:
        logname = path
    with open(logname, 'a+') as f:
        print(s, file=f)
        print(s)
    return

def distrubution_process(data, position, order) -> tuple:
    data = [x[position] for x in data]
    refs = {}
    for ele in data:
        if ele in refs.keys():
            refs[ele] += 1
        else:
            refs[ele] = 1

    if order == 'frequency':
        refs = sorted(refs.items(), key=lambda x: x[1])
    elif order == 'numerical':
        refs = sorted(refs.items(), key=lambda x: x[0])
    else:
        log_write(f'Wrong argument in function [{distrubution_process.__name__}]: order = {order} ')
        exit(1)

    subjects = [x[0] for x in refs]
    freqs = [x[1] for x in refs]
    return (subjects, freqs)

class Data_Unit:
    def __init__(self, data) -> None:
        if data[0] + 800 > 2400:
            return
        self.time = data[0] + 800
        self.date = data[1]
        self.mean = data[2]
        self.min = data[3]
        self.max = data[4]
        self.sd = data[5]
        self.tag = data[-1]
    def addZscore(self, z) -> None:
        self.zscore = z

class Data_Set:
    # global d312_20240101, d312_20240102, d312_20240103, d312_20240106, d312_20240107, d312_20240119, d312_20240329, d312_20240415, d312_20240417, d312_20240419
    d312_20240101 = [(1418, 1427, 1), (1555, 1605, 1), (1852, 1858, 1), (1903, 1907, 1), (1914, 1949, 1), (1957, 2032, 1), (2037, 2041, 1), (2047, 2053, 1), (2133, 2201, 1), (2206, 2233, 1)]
    d312_20240102 = [(1916, 1924, 1), (1925, 1934, 1), (1953, 2004, 1), (2010, 2039, 1), (2042, 2121, 1), (2128, 2132, 1), (2149, 2230, 1), (2237, 2323, 1), (2328, 2333, 1)]
    d312_20240103 = [(1436, 1441, 1), (1447, 1457, 1), (1459, 1509, 1), (1514, 1554, 1), (1600, 1640, 1), (1646, 1651, 1), (1911, 1916, 1), (1921, 1926, 1), (1928, 2005, 1), (2010, 2045, 1), (2047, 2052, 1), (2058, 2063, 1)]
    d312_20240106 = [(1539, 1543, 1), (1546, 1552, 1), (1559, 1636, 1), (1640, 1708, 1), (1736, 1758, 1), (1814, 1858, 2)]
    d312_20240107 = [(1500, 1504, 1), (1507, 1526, 1), (1537, 1554, 1), (1611, 1657, 3), (1701, 1748, 3), (1825, 1837, 1), (1844, 1854, 1), (1858, 1920, 1), (1922, 1925, 1), (1928, 1957, 1), (1958, 2001, 1), (2007, 2023, 1)]
    d312_20240119 = [(1941, 1948, 1), (1951, 2044, 1)]
    # d312_20240127 = [(1604, 1610, 1), (1616, 1656, 1), (2230, 2310, 1), (2315, 2323, 1)]
    d312_20240329 = [(1445, 1451, 1), (1457, 1542, 1)]
    d312_20240415 = [(1957, 2018, 2), (2034, 2110, 2)]
    d312_20240417 = [(1607, 1620, 2), (1631, 1652, 2)]
    d312_20240419 = [(1647, 1702, 2), (1713, 1803, 2), (1823, 1906, 2)]

    def __init__(self) -> None:
        self.others = []
        self.metro = []
        self.bus = []
        self.seg_others = []
        self.seg_metro = []
        self.seg_bus = []
    
    def add(self, dataunit) -> None:
        try:
            if dataunit.mean < 5:
                return
            if dataunit.tag == 0:
                tag_set = self.GetTagSet(dataunit.date)
                for (start, end, tag) in tag_set:
                    if dataunit.time - end < 3 and dataunit.time - end > 0:
                        return
                    elif start - dataunit.time < 3 and start - dataunit.time > 0:
                        return
                self.others.append(dataunit)
            elif dataunit.tag == 1:
                self.metro.append(dataunit)
            elif dataunit.tag == 2:
                self.bus.append(dataunit)
        except:
            return
    
    def GetTagSet(self, date) -> list:
        if date == '20240101':
            return self.d312_20240101
        elif date == '20240102':
            return self.d312_20240102
        elif date == '20240103':
            return self.d312_20240103
        elif date == '20240106':
            return self.d312_20240106
        elif date == '20240107':
            return self.d312_20240107
        elif date == '20240119':
            return self.d312_20240119
        elif date == '20240329':
            return self.d312_20240329
        elif date == '20240415':
            return self.d312_20240415
        elif date == '20240417':
            return self.d312_20240417
        elif date == '20240419':
            return self.d312_20240419
    
    def segment(self, length) -> None:
        print("Data unit: ", len(self.others), ' ', len(self.metro), ' ', len(self.bus))
        for k, dset in enumerate([self.others, self.metro, self.bus]):
            for i, dataunit in enumerate(dset):
                seg = [dataunit]
                for j in range(length - 1):
                    try:
                        if dataunit.date == dset[i + j + 1].date and dset[i + j].time + 1 == dset[i + j + 1].time:
                            seg.append(dset[i + j + 1])
                        else:
                            break
                    except:
                        break
                if len(seg) == length:
                    [self.seg_others, self.seg_metro, self.seg_bus][k].append(seg)
        print("Seg: ", len(self.seg_others), ' ', len(self.seg_metro), ' ', len(self.seg_bus))
        return
    
    def layout(self, testsize) -> list:
        train_set = []
        test_set = []
        random.seed(time.time())
        for k, dseg in enumerate([self.seg_others, self.seg_metro, self.seg_bus]):
            i = 0
            random.shuffle(dseg)
            for seg in dseg:
                seg_out = []
                seg_out.append([])
                for unit in seg:
                    seg_out[0].append(unit.mean)
                if i < testsize:
                    test_set.append([seg_out, seg[0].tag])
                    i += 1
                else:
                    train_set.append([seg_out, seg[0].tag])
        return train_set, test_set
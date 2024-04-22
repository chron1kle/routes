
import json, os, time
import numpy as np
import matplotlib.pyplot as plt
from basic_functions import *
##  UTC + 8

def GetTagSet(serial, date) -> list:
    try:
        return globals()[f'd{serial}_{date}']
    except KeyError:
        print(f'No transportation modes data.')
        return []

def resort(date, serial) -> None:
    data = loadData(serial, date)
    # data = 
    # save_labelled_data(data, date, serial)
    # log_write(f'Data on #{serial} {date} resorted according to time.')
    return sorted(data, key=lambda x: int(x[0]))

def labelling(date, serial) -> None:
    data = resort(date, serial)

    tag_set = GetTagSet(serial, date)

    # Marker 的意义是什么？
    # 用于确保下载的数据里包含了所有被记录的区间 -> 没有意义，删去 (solved)

    # Adjusting the 'Time' label
    for i, line in enumerate(data):
        data[i][0] = int(line[0][0:4])

    i = 0
    labelled = [0, 0, 0, 0, 0, 0]
    while i < len(data):
        mode = time_match(tag_set, data[i][0])
        data[i].append(mode)
        labelled[mode] += 1
        i += 1
    
    save_labelled_data(data, date, serial)

    return log_write(f"Successfully labelled. Labelled numbers: {labelled} .")

def remove_outliers(date, serial, sigma_multiplier) -> None:
    log_write(f"Performing function: {remove_outliers.__name__}")
    Ldata = loadLabelledData(serial, date)

    outList = []
    for tag in valid_tag:
        log_write(f'Tag {tag}:')
        data = np.array([x[2:5] for x in Ldata if x[-1] == tag])
        if len(data) == 0: continue
        for i in range(data.shape[1]):
            mean = np.mean(data[:, i])
            std = np.std(data[:, i])
            lower_threshold = mean - sigma_multiplier * std
            upper_threshold = mean + sigma_multiplier * std
            log_write(f'{i} : lower {lower_threshold} upper {upper_threshold}')
            for j, ele in enumerate(data[:, i]):
                if (lower_threshold > ele or ele > upper_threshold) and Ldata[j][-1] == tag and Ldata[j] not in outList:
                    log_write(f'Invalid data removed: {Ldata[j]}')
                    outList.append(Ldata[j])
    
    for ele in outList:
        Ldata.remove(ele)
    log_write(f'Removed {len(outList)} numbers of outliers data.')
    
    return save_cali_data(Ldata, date, serial)

# Kalman filter


# Removing Tag = -1 data
def removing_Invalid_data(data) -> list:
    poplist = []
    for line in data:
        if line[-1] == -1:
            poplist.append(line)
    for ele in poplist:
        data.remove(ele)
    log_write(f'Successfully removed {len(poplist)} number of Tag = -1 data.')
    return data

def alignment(date, serial):
    data = loadLabelledData(serial, date)
    for i in range(len(data)):
        if int(data[i][6]) == 0:
            data[i][6] = -1

    count = 0

    for i in range(len(data)):
        if int(data[i][6]) == -1:
            data[i][6] = 0
            count += 1
        else:
            break
    for i in range(len(data)):
        if int(data[-i-1][6]) == -1:
            data[-i-1][6] = 0
            count += 1
        else:
            break
    save_labelled_data(data, date, serial)
    log_write(f'{count} numbers of data beyond transportation period turned into blank.')
    return

def doLinearCali(head, tail, left_avg, right_avg, index, value) -> float:
    ratio = (index - head) / (tail - head)
    cali_value = (right_avg - left_avg) * ratio + left_avg
    
    return value - cali_value

def calibration(date, serial, running):

    if running == []:
        # static data
        print("Static data, no need for any calibration")
        return

    ####     parameters     ####
    sample_range = 60
    cali_range = range(2,3)

    log_write(f"Performing function: {calibration.__name__}")
    data = loadLabelledData(serial, date)
    head = running[0][0] - 800
    tail = running[-1][1] - 800
    headList = range(head-5, head+5)
    tailList = range(tail-5, tail+5)

    # pick 60 minutes before and after transportation period as calibration data

    ##### WRONG !!!!!!
    for i in range(len(data)):
        instant_time = int(data[i][0][0:4])
        if instant_time in headList:
            left = i - 30
        elif instant_time in tailList:
            right = i + 30

    log_write(f'Left: {left}, Right: {right}, head: {head}, tail: {tail}')

    for i in cali_range:  # just consider SVM_Mean for now
        left_sum = 0
        right_sum = 0
        
        try:
            cnt = 0
            while cnt < sample_range:
                if data[left][-1] in valid_tag:
                    left_sum += data[left][i]
                    cnt += 1
                left -= 1
            left_avg = left_sum / sample_range
        except IndexError:
            log_write(f'Not enough data for calibration from left. Actual number of data: {cnt}')
            left_avg = left_sum / cnt
        
        try:
            cnt = 0
            while cnt < sample_range:
                if data[right][-1] in valid_tag:
                    right_sum += data[right][i]
                    cnt += 1
                right += 1
            right_avg = right_sum / sample_range
        except IndexError:
            log_write(f'Not enough data for calibration from right. Actual number of data: {cnt}')
            right_avg = right_sum / cnt
        
        for j, line in enumerate(data):
            try:
                if int(line[0][0:4]) in range(head, tail+1):
                    data[j][i] = doLinearCali(head, tail, left_avg, right_avg, j, line[i])
            except IndexError as e:
                print(f'Error: {e}, len: {len(data)} j = {j}')
                exit(0)
        
    # Removing invalid (Tag = 1) data
    data = removing_Invalid_data(data)

    
                
    '''
    # calibrate Tag = 0 data
    for i in cali_range:
        calibrated_data = []
        j = 0
        while j < len(data):
            try:
                toBeCali = []
                while data[j + 1][0] - data[j][0] < 10:  # regard consecutive if the gap within 10 minutes
                    toBeCali.append(data[j][i])
                    j += 1
            except IndexError:
                continue
    '''
    

    #plt.scatter([x[0] for x in data], [x[2] for x in data])
    #print([x[2] for x in data])
    
    return save_cali_data(data, date, serial)

def BasicProcess(serial, dates) -> None:
    dataSet = LoadCatagory(serial, dates)


def DataVisualize(key, serial=None, date=None) -> None:
    if serial != None:
        data = loadLabelledData(serial, date)
        tag_set = GetTagSet(serial, date)
    x = []
    y = []
    colors = []
    for line in data:
        if line[0] > tag_set[0][0] - 60 - 800 and line[0] < tag_set[-1][1] + 60 - 800:
            x.append(line[0])
            y.append(line[key])
            if line[-1] == 0:
                colors.append('green')
            elif line[-1] == 1:
                colors.append('blue')
            elif line[-1] == 2:
                colors.append('red')
            # if line[key] > 200:
            #     print(line[0], ' ', line[5])
    plt.scatter(x, y, color = colors)
    return

def LabelAll(serial, dates) -> None:
    for date in dates:
        labelling(date, serial)
    return



def preprocess(date, serial, sigma_multiplier):  # standard workflow
    try:
        running = globals()[f'd{serial}_{date}']
    except KeyError:
        print(f'No transportation modes data.')
        running = []
    
    # resort(date, serial)  # sort the data in ascending order

    labelling(date, serial)  # tag those data within transportation period

    alignment(date, serial)   # mark out the uesless data within transportation period with value -1

    remove_outliers(date, serial, sigma_multiplier)  # removing outliers data

    calibration(date, serial, running) # calibrating data with transportation tag using static data
    return

if __name__ == '__main__':

    # parameters
    date = '20240327'
    serial = '312'
    
    # dates = ['20240420']

    sigma_multiplier = 3
    chunkLength = 20
    
    # LabelAll(serial, dates)

    log_write(f'------------------\n{time.ctime()}')

    # preprocess(date, serial, sigma_multiplier)

    exit(0)
    

    
# (solved) data within transportation should be labelled to -1 (invalid), in order not to interfere the purity of the training data
    
# Mark out those data in static state for calibration

# remove those unnormal data
    
# align the data
        
# decide the chunk length
    
# Kalman filter
    
# -------------------------------
    
# value, delta, 
import SQL_example as sqllib
import json, os, time
import numpy as np
import matplotlib.pyplot as plt
from basic_functions import feature_list, nodes_list, transportation_tag, valid_tag, loadData, loadLabelledData, loadCaliData, save_labelled_data, save_cali_data, time_match, log_write
##  UTC + 8

d312_20240101 = [(1418, 1427), (1555, 1605), (1852, 1858), (1903, 1907), (1914, 1949), (1957, 2032), (2037, 2041), (2047, 2053), (2133, 2201), (2206, 2233)]
d312_20240102 = [(1916, 1924), (1925, 1934), (1953, 2004), (2010, 2039), (2042, 2121), (2128, 2132), (2149, 2230), (2237, 2323), (2328, 2333)]
d312_20240106 = {1 : [(1539, 1543), (1546, 1552), (1559, 1636), (1640, 1708), (1736, 1758)], }


def resort(date, serial) -> None:
    data = loadData(serial, date)
    data = sorted(data, key=lambda x: int(x[0]))
    save_labelled_data(data, date, serial)
    log_write(f'Data on #{serial} {date} resorted according to time.')
    return 

def labelling(date, serial, running) -> None:
    data = loadLabelledData(serial, date)
    
    global marker
    for i in running:
        marker.append(0)

    i = 0
    labelled = 0
    while i < len(data):

        if time_match(running, data[i][0]):
            data[i].append(1)
            labelled += 1
        else:
            data[i].append(0)
        i += 1
    
    save_labelled_data(data, date, serial)
    s = ''
    for i in marker:
        if i == 0:
            s += ' ' + str(running[i]) + ' '
    if s == '':
        log_write(f"Successfully labelled. Totally {labelled} numbers of data labelled.")
    else:
        log_write(f"Error may have occurred. Missing data:\n{s}")
    return

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
    
    return save_labelled_data(Ldata, date, serial)

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
        return

    ####     parameters     ####
    sample_range = 60
    cali_range = range(2,3)

    log_write(f"Performing function: {calibration.__name__}")
    data = loadLabelledData(serial, date)
    head = running[0][0] - 800
    tail = running[-1][-1] - 800
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

    # Adjusting the 'Time' label
    for i, line in enumerate(data):
        data[i][0] = int(line[0][0:4])
                
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



def preprocess(date, serial, running, sigma_multiplier):  # standard workflow
    
    resort(date, serial)  # sort the data in ascending order

    labelling(date, serial, running)  # tag those data within transportation period

    alignment(date, serial)   # mark out the uesless data within transportation period with value -1

    remove_outliers(date, serial, sigma_multiplier)  # removing outliers data

    calibration(date, serial, running) # calibrating data with transportation tag
    return

if __name__ == '__main__':

    # parameters
    date = '20240106'
    serial = '312'

    logname = 'MajorLog.log'
    sigma_multiplier = 3
    chunkLength = 20
    marker = []
    try:
        running = globals()[f'd{serial}_{date}']
    except KeyError:
        print(f'No transportation modes data.')
        running = []

    log_write(f'------------------\n{time.ctime()}')

    preprocess(date, serial, running, sigma_multiplier)

    exit(0)
    

    
# (solved) data within transportation should be labelled to -1 (invalid), in order not to interfere the purity of the training data
    
# Mark out those data in static state for calibration

# remove those unnormal data
    
# align the data
        
# decide the chunk length
    
# Kalman filter
    
# -------------------------------
    
# value, delta, 
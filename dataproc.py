import SQL_example as sqllib
import json, os, time
##  UTC + 8

'''
 -1: invalid
 0: static or walking
 1: MTR
'''



d312_20240101 = [(1418, 1427), (1555, 1605), (1852, 1858), (1903, 1907), (1914, 1949), (1957, 2032), (2037, 2041), (2047, 2053), (2133, 2201), (2206, 2233)]
d312_20240102 = [(1916, 1924), (1925, 1934), (1953, 2004), (2010, 2039), (2042, 2121), (2128, 2132), (2149, 2230), (2237, 2323), (2328, 2333)]

d312_20240308 = []
d312_20240309 = []
d312_20240307 = []
d312_20240306 = []

def save_labelled_data(d, date, serial) -> None:
    filename = f'data\\labelled-{serial}-{date}.json'
    try:
        with open(filename, 'w+') as f:
            json.dump(d, f)
        log_write(f'\nLabelled data successfully saved to {filename}\n')
    except Exception as e:
        log_write(f'\nFailed to store labelled data. Error: {e}\n')
    return

def time_match(running, record) -> bool:
    global marker
    instant = int(record[0:4])
    for i, (start, end) in enumerate(running):
        if start - 800 <= instant and end - 800 >= instant:
            marker[i] = 1
            return True
        else:
            continue
    return False

def log_write(s) -> None:
    global logname
    with open(logname, 'a') as f:
        print(s, file=f)
        print(s)
    return
        
def labelling(date, serial, running) -> None:
    data = sqllib.loadLabelledData(serial, date)
    
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

def removeZero(date, serial):
    data = sqllib.loadLabelledData(serial, date)
    count = 0
    for i in range(len(data)):
        if int(data[i][2]) < 10 or int(data[i][3]) < 10 or int(data[i][4]) < 10:  # SVM_mean = 0 (Invalid)
            data[i][6] = -1
            count += 1
    log_write(f'{count} numbers of invalid (zero) data found.')
    return save_labelled_data(data, date, serial)

def alignment(date, serial):
    data = sqllib.loadLabelledData(serial, date)
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

def calibration(date, serial, running):
    data = sqllib.loadLabelledData(serial, date)
    head = running[0][0]
    tail = running[-1][-1]
    headList = range(head-5, head+5)
    tailList = range(tail-5, tail+5)

    for i in range(len(data)):
        instant = int(data[i][0][0:4])
        if instant in headList:
            left = i - 30
        elif instant in tailList:
            right = i + 30

    mean_sum = 0
    min_sum = 0
    max_sum = 0
    cnt = 0

    for i in range(0, left):
        if data[i][-1] == -1:
            continue
        mean_sum += data[i][2]
        min_sum += data[i][3]
        max_sum += data[i][4]
        cnt += 1

    for i in range(right, len(data)):
        if data[i][-1] == -1:
            continue
        mean_sum += data[i][2]
        min_sum += data[i][3]
        max_sum += data[i][4]
        cnt += 1
    
    save_labelled_data(data, date, serial)
    return

def resort(date, serial):
    data = sqllib.loadData(serial, date)
    data = sorted(data, key=lambda x: int(x[0]))
    save_labelled_data(data, date, serial)
    log_write(f'Data on #{serial} {date} resorted according to time.')
    return 

def preprocess(date, serial, running):  # standard workflow
    
    resort(date, serial)  # sort the data in ascending order

    labelling(date, serial, running)  # tag those data within transportation period

    alignment(date, serial)   # mark out the uesless data within transportation period with value -1

    removeZero(date, serial)  # mark out data with invalid value '0'
    return

if __name__ == '__main__':

    # override
    logname = 'MajorLog.log'
    date = '20240102'
    serial = '312'
    marker = []
    running = globals()[f'd{serial}_{date}']

    log_write(f'------------------\n{time.ctime()}')

    preprocess(date, serial, running)

    exit(0)
    

    
# (solved) data within transportation should be labelled to -1 (invalid), in order not to interfere the purity of the training data
    
# Mark out those data in static state for calibration

# remove those invalid data
    
# align the data
        
# decide the chunk length
    
# Kalman filter
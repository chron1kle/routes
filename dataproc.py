import SQL_example as sqllib
import json, os, time
##  UTC + 8

d612_20240101 = [(1418, 1427), (1555, 1605), (1852, 1858), (1903, 1907), (1914, 1949), (1957, 2032), (2037, 2041), (2047, 2053), (2133, 2201), (2206, 2233)]


def save_labelled_data(d, date, serial) -> None:
    filename = f'data\\labelled-{serial}-{date}.json'
    try:
        with open(filename, 'w+') as f:
            json.dump(d, f)
        log_write(f'\nLabelled data successfully saved to {filename}\n')
    except Exception as e:
        print(f'\nFailed to store labelled data. Error: {e}\n')

def time_match(running, record) -> bool:
    #print(int(record[0:4]))
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

if __name__ == '__main__':

    # override
    logname = 'dataproc.log'
    date = '20240101'
    serial = '312'
    running = d612_20240101
    log_write(f'------------------\n{time.ctime()}')
    
    original_length = len(running)

    marker = []
    for i in running:
        marker.append(0)

    data = sqllib.loadData(serial, date)

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
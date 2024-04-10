import json, os, time, random
from basic_functions import *

folder_path = 'data\\'

seg_length = 3
offset = 5
training_set_length = 40

dest_name = f'{folder_path}segs_{seg_length}_{offset}.json'

def isConsecutive(seg, deviceID, date):
    # check if consecutive
    start = seg[0][0]
    end = seg[-1][0]
    label = seg[0][-1]
    for unit in seg[1:]:
        if unit[0] == start + 1 and unit[-1] == label:
            start += 1
        else:
            return False

    # ensure to keep distance with dynamic data
    if label == 0:
        records = globals()[f'd{deviceID}_{date}']
        for unit in records:
            if start > unit[1] + offset or end < unit[0] - offset:
                continue
            else:
                return False

    return True

def segmentize():
    for filename in os.listdir(folder_path):
        if filename[0:4] != 'cali':
            continue

        with open(f'{folder_path}{filename}', 'r') as f:
            data = json.load(f)
            deviceID = filename[5:8]
            date = filename[9:17]
        log_write(f'Performing segmentization on {filename}')
        segs = []
        for i in range(len(data)):
            try:
                seg = data[i : i + seg_length]
                if isConsecutive(seg, deviceID, date):
                    segs.append([[ x[2:-1] for x in seg ], seg[0][-1]])
            except Exception as e:
                print(f'Exception: {e}')
                continue
        if segs != []:
            save_seg_data(segs, dest_name)
    return

def select_training_set():
    data = load_seg_data(seg_length, offset)
    testing_set = []
    random.seed(int(time.time()))
    for i in range(training_set_length):
        rd = random.randint(0, len(data) - 1)
        testing_set.append(data.pop(rd))
    return save_train_data(data, testing_set, seg_length, offset)


if __name__ == '__main__':
    with open(dest_name, 'w') as f:
        pass
    segmentize()
    select_training_set()
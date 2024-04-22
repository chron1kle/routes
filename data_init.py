import json, os, time, random
from basic_functions import *

folder_path = 'data\\'

seg_length = 3
offset = 5
training_set_length = 40

flag = ["model", "confi"][0]

if flag == "model":
    dest_name = f'{folder_path}segs_{seg_length}_{offset}.json'
elif flag == "confi":
    dest_name = f'{folder_path}segsConfi_{seg_length}_{offset}.json'
else:
    print(f'Wrong flag: {flag}')
    exit(1)

def isConsecutive(seg, deviceID, date):
    # check if consecutive
    # print(seg)
    # exit(0)
    for chunk in seg:
        if chunk[2] < 0:
            return "abandon"
    start = seg[0][0]
    end = seg[-1][0]
    label = seg[0][-1]
    for unit in seg[1:]:
        if unit[0] == start + 1:
            start += 1
        else:
            return "time"
    for unit in seg[1:]:
        if unit[-1] == label:
            pass
        else:
            return "tag"

    # ensure to keep distance with dynamic data
    if label == 0:
        records = globals()[f'd{deviceID}_{date}']
        for unit in records:
            if start > unit[1] + offset or end < unit[0] - offset:
                continue
            else:
                return "unsafe"

    return "consecutive"

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
                
                if len(seg) != seg_length:
                    continue
                elif isConsecutive(seg, deviceID, date) == "consecutive":
                    # for j in range(len(seg)):
                    #     seg[j] = [ [x] for x in seg[j]]
                    # seg_1d = []
                    # for chunk in [ x[2:-1] for x in seg ]:
                    #     for d in chunk:
                    #         seg_1d.append(d)
                    if flag == "confi":
                        segs.append([[ x[0:2] for x in seg ], [ x[2:-1] for x in seg ], seg[0][-1]])
                    else:
                        segs.append([[ x[2:-1] for x in seg ], seg[0][-1]])
                # elif isConsecutive(seg, deviceID, date) == "tag":
                #     if flag == "confi":
                #         segs.append([[ x[0:2] for x in seg ], [ x[2:-1] for x in seg ], 0])
                #     else:
                #         segs.append([[ x[2:-1] for x in seg ], 0])
            except Exception as e:
                print(f'Exception: {e}')
                continue
        if segs != []:
            save_seg_data(segs, dest_name)
    return

def randomize():
    segs = []
    data = load_seg_data(seg_length, offset)
    tag_classifier = [[], [], [], []]
    for seg in data:
        tag_classifier[seg[-1]].append(seg[0])
    random.seed(time.time())
    for i in range(150):
        new_seg = [[], 0]
        r = random.randint(1, seg_length - 1)
        for _ in range(r):
            new_seg[0].append(random.choice(random.choice(tag_classifier[1])))
        for _ in range(seg_length - r):
            new_seg[0].append(random.choice(random.choice(tag_classifier[2])))
        segs.append(new_seg)
    for i in range(150):
        new_seg = [[], 0]
        r = random.randint(1, seg_length - 1)
        for _ in range(r):
            new_seg[0].append(random.choice(random.choice(tag_classifier[1])))
        for _ in range(seg_length - r):
            new_seg[0].append(random.choice(random.choice(tag_classifier[3])))
        segs.append(new_seg)
    for i in range(150):
        new_seg = [[], 0]
        r = random.randint(1, seg_length - 1)
        for _ in range(r):
            new_seg[0].append(random.choice(random.choice(tag_classifier[2])))
        for _ in range(seg_length - r):
            new_seg[0].append(random.choice(random.choice(tag_classifier[3])))
        segs.append(new_seg)
    # print(segs[0], segs[1])
    save_seg_data(segs, dest_name)
    

if __name__ == '__main__':
    print(f'flag: {flag}')
    time.sleep(1)
    with open(dest_name, 'w') as f:
        pass
    segmentize()
    # randomize()
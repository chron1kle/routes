import SQL_example as sqllib
import json, os
##  UTC + 8

d612_20240101 = [(1418, 1427), (1555, 1605), (1852, 1858), (1903, 1907), (1914, 1949), (1957, 2032), (2037, 2041), (2047, 2053), (2133, 2201), (2206, 2233)]


def prt_results() -> None:

    return

def svm_count():

    return

def time_match(actual, record) -> bool:
    if str(actual - 800) == record[0:4]:
        return True
    else:
        return False


if __name__ == '__main__':

    # override
    date = '20240101'
    serial = '312'
    running = 'd' + serial + '_' + date

    data = sqllib.loadData(serial, date)

    i = 0
    while i < len(data):
        itr = running.pop(0)
        length = itr[1] - itr[0] + 1

        if time_match(itr[0], data[i][0]):
            pass
        else:
            pass

        for j in range(length):
            pass


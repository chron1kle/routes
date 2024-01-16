
'''

This py script is for data fetching, from server to your own host.
You should first specify the target device ID and the exact date of the data you intend to fetch.

For example, if you want to get the data from device no.312 on the date 2024-01-03,
then you should modify the variables, which are at the bottom part of this script, {date} into 20240103, and {node} into 2.

After running the script, you will get all the data for device 312 on the day 2024-01-03, which will be stored in the folder named {data} under your current directory,
and the json file will be named as 312-20240103.json, under the folder {data}. 
You could modify the folder name and file-naming rules into whatever you like by modifying the codes inside functions {storeData} and {loadData}.

Please be reminded that all the raw data you fetched are labelled in the standard UTC time, which is the London local time. 
Since Hongkong is in UTC+8, for example, if you wanna get the data in HKT 14:10:00 pm, you should find the one labelled in the time of {061000}.

'''

import mysql.connector, mysql.connector.pooling
from mysql.connector import errorcode
import os, json

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

def data_fetch(feature_list, node, specified_query) -> list:
    rawPool = ConPool(target_DB="sensors_data", file_name="SQL_credential.txt")
    
    feature_str = ", ".join(feature_list)
    sql = "SELECT %s from nodesdatapool WHERE NodeID = %s %s" % (feature_str, node, specified_query) # specipfy the num of record lines to fetch
    results = rawPool.selectAll(sql)
    print(f'\n{len(results)} of lines of records found successfully\n')
        
    return results

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
        print(f'\nData on {date} successfully loaded.\n')
    except Exception as e:
        print(f'\nFailed. Error: {e}\n')
    return data

class ConPool():
    def __init__(self,target_DB = None, file_name = "database_pass"):
           
        connect_para = {}
        with open(file_name) as paraFile:
            for line in paraFile:
                connect_para[line.split()[0]] = line.split()[1]    
        
        dbconfig = {  
                "user": connect_para['user'],  
                "password": connect_para['pass'],  
                "host": connect_para['host'],  
#                 "charset": "utf8"  
            }
        print(dbconfig)
        try:
            self.mysql_pool = mysql.connector.pooling.MySQLConnectionPool(pool_name="mysql_pool",pool_size = 10,pool_reset_session=True, **dbconfig)
            if target_DB is not None:
                self.selectDB(target_DB)
            print("Connection established")
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Something is wrong with the user name or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database does not exist")
            else:
                print(err)

    def selectDB(self, target_DB):
        self.mysql_pool.set_config(**{"database":target_DB})


    def execute_sql(self, sql):
        try:
            self.conn = self.mysql_pool.get_connection()
            self.cursor = self.conn.cursor(buffered=True)
            
            
            results = self.cursor.execute(sql)
            return results
            
        except mysql.connector.Error as err:
            print(err)
        
    def connClose(self):
        self.cursor.close()
        self.conn.close()
        
               
    def selectOne(self, sql):
        self.execute_sql(sql)
        results = self.cursor.fetchone()
        self.connClose()
        
        return results
    
    def selectAll(self, sql):
        self.execute_sql(sql)
        results = self.cursor.fetchall()
        self.connClose()
        
        return results
                   
    
    def getConn(self):
        self.conn = self.mysql_pool.get_connection()
        self.cursor = self.conn.cursor(buffered=True)
        
        return self.cursor


'''
!!!  Please do not modify the codes above unless necessary  !!!
'''

if __name__ == '__main__':

    # These are the arguments you have to specify or modify whenever you run this script

    date = '20240101'
    node = 2  # 0 for 305, 1 for 306, 2 for 312

    # These are the arguments you could but not have to modify, unless necessary

    ascending_or_descending = 'ASC'  # specifying the ordering of the data you will fetch. ASC for ascending, and DESC for descending
    ordering = 'id'  # specifying which feature will be applied for ordering. You can chanee to other features which are included in the list {feature_list}, which is at the top of this script
    feature_list = ['Time', 'Date', 'SVM_mean', 'SVM_min', 'SVM_max', 'SVM_SD']   # specifying the features you wanna fetch. In our project, only SVM is valid for our model training (I guess).


    '''
    !!!  Please do not modify the codes below unless necessary  !!!
    '''

    specified_query = f'AND Date = {date} ORDER BY {ordering} {ascending_or_descending}'

    raw_data = data_fetch(feature_list, nodes_list[node], specified_query)
    storeData(raw_data, nodes_list[node], date)
import os 
import sys
import json 

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URI = os.getenv("MONGO_DB_URI")
print(f"MONGO_DB_URI: {MONGO_DB_URI}")

import certifi
ca = certifi.where()

import pandas as pd
import pymongo
import numpy as np
from Networksecurity.exception.exception import NetworkSecurityException    
from Networksecurity.logging.logger import logging

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def cv_to_json(self,file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def insert_data_mongodb(self,records,collection,database):
        try:
            self.database=database
            self.collection=collection
            self.records=records

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URI, tlsCAFile=ca)
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise NetworkSecurityException(e, sys)
if __name__ == "__main__":
    FILE_PATH = r"C:\Users\ASUS\Desktop\project\phishing.csv"
    DATABASE = "Networksecurity"
    COLLECTION = "phishing"

    networkobj = NetworkDataExtract()
    records = networkobj.cv_to_json(file_path=FILE_PATH)   # fixed method name
    no_of_records = networkobj.insert_data_mongodb(
        records=records,
        collection=COLLECTION,
        database=DATABASE
    )
    print(f"Number of records inserted: {no_of_records}")

import sys
import os
from src.exception import customException
from src.logger import logging

import pandas as pd

from dataclasses import dataclass

@dataclass
class DataCleaningConfig:
    cleaning_file_path:str =os.path.join("artifacts","cleaned_data.csv")

class DataCleaning:
    def __init__(self):
        self.data_cleaning_config=DataCleaningConfig()
    
    def data_cleaning(self):
        try:
            df=pd.read_csv('notebooks/data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
            logging.info("read the csv before cleaning and set to clean.")
            df = df.drop(columns=["customerID"])
            df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"})
            df["TotalCharges"] = df["TotalCharges"].astype(float)

            os.makedirs(os.path.dirname(self.data_cleaning_config.cleaning_file_path),exist_ok=True)
            df.to_csv(self.data_cleaning_config.cleaning_file_path,index=False,header=True)
            
            logging.info("the data is cleaning is done and saved to folder.")
            return (
                self.data_cleaning_config.cleaning_file_path
            )
        except Exception as e:
            raise customException(e,sys)
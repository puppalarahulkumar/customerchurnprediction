import os
import sys
from src.exception import customException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_cleaning import DataCleaning
from src.components.data_transformation import DataTransformation,DataTransfomationConfig

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")
    raw_data_path:str=os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):   
        logging.info("entered the data ingeston method")
        try:
            data_cleaning=DataCleaning()
            cleaned_data_path=data_cleaning.data_cleaning()
            df=pd.read_csv(cleaned_data_path)
            logging.info("read the dataset as frame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            train_set,test_set=train_test_split(df,random_state=44,test_size=0.20)
            train_set.to_csv(self.ingestion_config.train_data_path)
            test_set.to_csv(self.ingestion_config.test_data_path)

            logging.info("ingestion of data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise customException(e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    train_path,test_path=obj.initiate_data_ingestion()

    data_transform=DataTransformation()
    traina_arr,test_arr=data_transform.initiate_data_transformation(train_path,test_path)

    


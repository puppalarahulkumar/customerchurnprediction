from src.exception import customException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__=="__main__":
    obj=DataIngestion()
    train_path,test_path=obj.initiate_data_ingestion()

    data_transform=DataTransformation()
    train_arr,test_arr,_=data_transform.initiate_data_transformation(train_path,test_path)

    model_training=ModelTrainer()
    print(model_training.initiate_model_trainer(train_arr,test_arr))
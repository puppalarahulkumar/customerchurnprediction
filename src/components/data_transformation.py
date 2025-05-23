import os
import sys
from dataclasses import dataclass
import numpy as np

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from src.exception import customException
from src.logger import logging
from src.utils import save_object


# sample oversampling
from imblearn.over_sampling import RandomOverSampler



@dataclass
class DataTransfomationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransfomationConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_columns=['tenure','MonthlyCharges','TotalCharges']
            categorical_columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
       'PaperlessBilling', 'PaymentMethod']
            
            num_pipeline=Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy='median')),
                    ("Scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="most_frequent")),
                    ("Encoding",OneHotEncoder(handle_unknown='ignore'))
                ]
            )
            logging.info(' num and cat pipeline is created! ')
            preprocessor=ColumnTransformer(
                [
                    ('num pipeline',num_pipeline,numerical_columns),
                    ('cat pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
            


        except Exception as e:
            raise customException(e,sys)
      
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("reading the dataset in transformation is done")

            preprocessor_obj=self.get_data_transformer_object()

            target_feature="Churn"

            input_feature_train_df=train_df.drop(columns=[target_feature],axis=1)
            target_feature_train_df=train_df[target_feature].replace({'Yes':1,'No':0}).astype(int)

            input_feature_test_df=test_df.drop(columns=[target_feature],axis=1)
            target_feature_test_df=test_df[target_feature].replace({'Yes':1,'No':0}).astype(int)

            ros=RandomOverSampler()
            input_feature_train_df,target_feature_train_df=ros.fit_resample(input_feature_train_df,target_feature_train_df)
            input_feature_test_df,target_feature_test_df=ros.fit_resample(input_feature_test_df,target_feature_test_df)

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)


            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            # new_train_arr=pd.DataFrame(train_arr)
            # new_test_arr=pd.DataFrame(test_arr)
            # new_train_arr.to_csv("trained_arr.csv")
            # new_test_arr.to_csv("tested_arr.csv")

            save_object(
                self.data_transformation_config.preprocessor_obj_file_path,
                preprocessor_obj
            )

            return(train_arr,test_arr,self.data_transformation_config)
        
        except Exception as e:
            raise customException(e,sys)
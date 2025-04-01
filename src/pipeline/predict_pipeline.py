from src.exception import customException
from src.logger import logging

import sys
import pandas as pd

from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass 
    def predict(self,features):
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/preprocessor.pkl'
            model=load_object(model_path)
            preprocessor=load_object(preprocessor_path)
            data_scaled=preprocessor.transform(features)
            print(data_scaled)
            pred=model.predict(data_scaled)
            return pred
        except Exception as e:
            raise customException(e,sys)
    

class CustomData:
    def __init__(self,gender:str,SeniorCitizen:int,Partner:str,Dependents:str,tenure:int,PhoneService:str,MultipleLines:str,
                 InternetService:str,OnlineSecurity:str,OnlineBackup:str,DeviceProtection:str,TechSupport:str,
                 StreamingTV:str,StreamingMovies:str,Contract:str,PaperlessBilling:str,PaymentMethod:str,MonthlyCharges:float,TotalCharges:float):
        self.gender:str=gender
        self.SeniorCitizen:int=SeniorCitizen
        self.Partner:str=Partner
        self.Dependents:str=Dependents
        self.tenure:int=tenure
        self.PhoneService:str=PhoneService
        self.MultipleLines:str=MultipleLines
        self.InternetService:str=InternetService
        self.OnlineSecurity:str=OnlineSecurity
        self.OnlineBackup:str=OnlineBackup
        self.DeviceProtection:str=DeviceProtection
        self.TechSupport:str=TechSupport
        self.StreamingTV:str=StreamingTV
        self.StreamingMovies:str=StreamingMovies
        self.Contract:str=Contract
        self.PaperlessBilling:str=PaperlessBilling
        self.PaymentMethod:str=PaymentMethod
        self.MonthlyCharges:float=MonthlyCharges
        self.TotalCharges:float=TotalCharges
    
    def get_data_as_frame(self):
        try:
            custom_data_input_dist = {
            "gender": [self.gender],
            "SeniorCitizen": [self.SeniorCitizen],
            "Partner": [self.Partner],
            "Dependents": [self.Dependents],
            "tenure": [self.tenure],
            "PhoneService": [self.PhoneService],
            "MultipleLines": [self.MultipleLines],
            "InternetService": [self.InternetService],
            "OnlineSecurity": [self.OnlineSecurity],
            "OnlineBackup": [self.OnlineBackup],
            "DeviceProtection": [self.DeviceProtection],
            "TechSupport": [self.TechSupport],
            "StreamingTV": [self.StreamingTV],
            "StreamingMovies": [self.StreamingMovies],
            "Contract": [self.Contract],
            "PaperlessBilling": [self.PaperlessBilling],
            "PaymentMethod": [self.PaymentMethod],
            "MonthlyCharges": [self.MonthlyCharges],
            "TotalCharges": [self.TotalCharges]
            }

            return pd.DataFrame(custom_data_input_dist)
        
        except Exception as e:
            raise customException(e,sys)
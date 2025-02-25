import os 
import sys
import dill
import pandas as pd
from src.exception import customException
from src.logger import logging
from sklearn.metrics import accuracy_score

def save_object(file_path,obj):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise customException(e,sys)

def evaluate_models(x_train,y_train,x_test,y_test,models):
    try:
        report={}
        for i in range(len(list(models.values()))):
            model=list(models.values())[i]

            model.fit(x_train,y_train)

            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)

            train_model_score=accuracy_score(y_train,y_train_pred)
            test_model_score=accuracy_score(y_test,y_test_pred)

            report[list(models.keys())[i]]=test_model_score

        return report
    except Exception as e:
        raise customException(e,sys)



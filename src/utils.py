import os 
import sys
import dill
import pandas as pd
from src.exception import customException
from src.logger import logging
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, recall_score, roc_auc_score

def save_object(file_path,obj):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise customException(e,sys)

def evaluate_models(x_train,y_train,x_test,y_test,models,params):
    try:
        report={}
        for i in range(len(list(models.values()))):
            model=list(models.values())[i]

            param=params[list(models.keys())[i]]
            
            randomcv=RandomizedSearchCV(model,param,cv=5,verbose=2,n_jobs=-1)
            randomcv.fit(x_train,y_train)
            
            model.set_params(**randomcv.best_params_)
            model.fit(x_train,y_train)

            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)

            train_model_score=accuracy_score(y_train,y_train_pred)
            test_model_score=accuracy_score(y_test,y_test_pred)


            f1 = f1_score(y_test, y_test_pred)
            recall = recall_score(y_test, y_test_pred)


            print(f"Accuracy: {test_model_score}")
            print(f"F1 Score: {f1}")
            print(f"Recall: {recall}")


            
            report[list(models.keys())[i]]=test_model_score

        return report
    except Exception as e:
        raise customException(e,sys)


def load_object(file_path):
    try:
        with open(file_path,'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise customException(e,sys)
import numpy as np
import pandas as pd
import os
import sys

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# performance metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from src.utils import save_object
from src.logger import logging
from dataclasses import dataclass
from src.exception import customException
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("initiating the train and test split")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "Decision Tree Classifier":DecisionTreeClassifier(),
                "Random Forest Classifier":RandomForestClassifier(),
                "KNeighborsClassifier":KNeighborsClassifier(),
                "SVC":SVC(),
                "LogisticRegression":LogisticRegression()
            }

            params = {
                    "Decision Tree Classifier": {
                        "criterion": ["gini", "entropy"],
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4]
                    },
                    "Random Forest Classifier": {
                        "n_estimators": [50, 100, 200],
                        "criterion": ["gini", "entropy"],
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4]
                    },
                    "KNeighborsClassifier": {
                        "n_neighbors": [3, 5, 7, 9],
                        "weights": ["uniform", "distance"],
                        "metric": ["euclidean", "manhattan", "minkowski"]
                    },
                    "SVC": {
                        "C": [0.1, 1, 10, 100],
                        "kernel": ["linear", "rbf", "poly", "sigmoid"],
                        "gamma": ["scale", "auto"]
                    },
                    "LogisticRegression": {
                        "penalty": ["l1", "l2", "elasticnet", None],
                        "C": [0.1, 1, 10, 100],
                        "solver": ["lbfgs", "liblinear", "saga"],
                        "max_iter": [100, 200, 300]
                    }
                }


            models_report:dict=evaluate_models(x_train=X_train,x_test=X_test,y_train=y_train,y_test=y_test,models=models,params=params)

            best_model_score=max(models_report.values())

            best_model_name=list(models_report.keys())[
                list(models_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise customException("no best model found")
            logging.info("the best model is found on both training and testing data")

            save_object(self.model_trainer_config.trained_model_file_path,best_model)

            predicted=best_model.predict(X_test)
            print(predicted)
            accuracy=accuracy_score(y_test,predicted)

            return accuracy
        
        except Exception as e :
            raise customException(e,sys)
from src.exception import customException
from src.logger import logging
from flask import Flask,Response,request,render_template
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
app=application

# route to home page
# @app.route('/')
# def index():
#     return render_template("index.html")

@app.route('/',methods=['GET','POST'])
def predict_data():
    if request.method == "GET":
        return render_template('predict.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            SeniorCitizen=request.form.get('SeniorCitizen'),
            Partner=request.form.get('Partner'),
            Dependents=request.form.get('Dependents'),
            tenure=request.form.get('tenure'),
            PhoneService=request.form.get('PhoneService'),
            MultipleLines=request.form.get('MultipleLines'),
            InternetService=request.form.get('InternetService'),
            OnlineSecurity=request.form.get('OnlineSecurity'),
            OnlineBackup=request.form.get('OnlineBackup'),
            DeviceProtection=request.form.get('DeviceProtection'),
            TechSupport=request.form.get('TechSupport'),
            StreamingTV=request.form.get('StreamingTV'),
            StreamingMovies=request.form.get('StreamingMovies'),
            Contract=request.form.get('Contract'),
            PaperlessBilling=request.form.get('PaperlessBilling'),
            PaymentMethod=request.form.get('PaymentMethod'),
            MonthlyCharges=request.form.get('MonthlyCharges'),
            TotalCharges=request.form.get('TotalCharges')
        )
        pred_df=data.get_data_as_frame()
        print(pred_df)
        final = "Invalid Prediction"
        pipeline=PredictPipeline()
        results=pipeline.predict(features=pred_df)
        print(results[0])
        if results[0]==1.0:
            final="The Customer is not likely to Churn"
        elif results[0]==0.0:
            final="The Customer is likely to Churn"

        return render_template('predict.html',results=final)


if __name__=="__main__":


    app.run(host="0.0.0.0",debug=True)
    print("app is running")
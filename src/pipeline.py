import mlflow
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric
from preprocess import process_data
from train import train,load_model,save_model,handle_training_results
import pandas as pd
import argparse

# Start MLflow run
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_registry_uri("sqlite:///mlflow.db")
parser = argparse.ArgumentParser(description="Train a spam classification model.")
parser.add_argument("--model", type=str, required=True, choices=["logistic_regression", "random_forest"],
                      help="Model to train: 'logistic_regression' or 'random_forest'")

class SpamDetectionPipeline:
  def __init__(self,data_flow):
    self.data_flow =data_flow
    pass
  def run(self,retrain =False,model_name='logistic_regression'):

    while self.data_flow != None and len(self.data_flow) > 0:
      
      #1. Data Ingestion
      next_data = self.data_flow.pop()

      #2. Preprocessing
      processed_data = process_data(next_data['data'], next_data['suffix'])

      #3. Training
      #!!!!IMPORTANTE!!!! Na hora de Retreinar tem que Continuar o Treinamento não substituir!
      #!!!! IMPORTANTE !!!! Separar Training de Evaluation
      
      training_results =train(processed_data,model_name,load_model(model_name))
      save_model(model_name)
      #4. Evaluation
      handle_training_results(training_results)
      

      #5 Drift Analysis Setup

      #6 Definir se deve retreinar

      


      if retrain==False:
        
        break
    #7 Deploy Flask Docker
      
    



if __name__ == "__main__":
  with mlflow.start_run():
    data1 = pd.read_csv('../data/spam1.csv', encoding='latin-1')
    data2 = pd.read_csv('../data/spam2.csv', encoding='latin-1')
    data_flow = [{"data":data2,"suffix":"2"},{"data":data1,"suffix":"1"}]
    pipeline = SpamDetectionPipeline(data_flow=data_flow)
    args = parser.parse_args()
    pipeline.run(True,args.model)
import mlflow
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric
from preprocess import process_data
from train import train,save_model,evaluate
from app import update_production_model
import pandas as pd
import argparse
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature


parser = argparse.ArgumentParser(description="Train a spam classification model.")
parser.add_argument("--model", type=str, required=True, choices=["xgboost", "random_forest"],
                      help="Model to train: 'xgboost' or 'random_forest'")

FORCE_DRIFT = False
EXPERIMENT_NAME = 'Diabetes-Prediction'
MODEL_NAME = 'Diabetes-Model'

class DiabetesDetectionPipeline:
  def __init__(self,data_flow,mlflow_experiment_id):
    self.data_flow = data_flow
    self.mlflow_experiment_id = mlflow_experiment_id

  def check_drift(self,training_data,new_data):
      if training_data.empty:
        return True
      
      if FORCE_DRIFT:
        print("🚨 Drift detectado! Adicionando dados e retreinando o modelo.")
        return FORCE_DRIFT

      drift_report = Report(metrics=[DatasetDriftMetric()])
      drift_report.run(
          reference_data=training_data, 
          current_data=new_data
      )
      drift = drift_report.as_dict()['metrics'][0]['result']['dataset_drift']
      if drift:
        print("🚨 Drift detectado! Adicionando dados e retreinando o modelo.")

      return drift

  def get_input_example(self):
    return self.training_data.iloc[0:1]  # Exemplo: primeira linha dos dados de treino

  def save_new_best(self,training_results,old_roc,new_roc):
    
    mlflow.sklearn.log_model(training_results.model, MODEL_NAME)
    mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model",name=MODEL_NAME)
    
    save_model(training_results.model, "best")  # Atualiza o modelo em produção
    print(f"🎉 Novo melhor ROC: {new_roc:.4f} (anterior: {old_roc:.4f})")


  def save_metrics(self,evaluation_results):
    mlflow.log_metric("accuracy", evaluation_results.accuracy)
    mlflow.log_metric("precision", evaluation_results.precision)
    mlflow.log_metric("recall", evaluation_results.recall)
    mlflow.log_metric("f1_score", evaluation_results.f1_score)
    mlflow.log_metric("roc_auc_score", evaluation_results.roc_auc_score)

  def check_if_new_winner(self,evaluation_results, training_results):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc_score DESC"],
        max_results=1
    )

    if len(runs) > 0 and len(runs[0].data.metrics.items()) > 0:
      best_roc = runs[0].data.metrics["roc_auc_score"]
    else:
      best_roc = 0

    new_roc = evaluation_results.roc_auc_score
    
    if new_roc > best_roc:
        self.save_new_best(training_results,best_roc,new_roc)
    else:
        print(f"⚠️  ROC atual ({new_roc:.4f}) não supera o melhor ({best_roc:.4f})")
    
    self.save_metrics(evaluation_results)
    return True

  def update_production(self):
    client = MlflowClient()
    client.transition_model_version_stage(
    version=1,
    name=MODEL_NAME,
    stage="Production"
    )
    return True


  def run(self,retrain =False,model_name='xgboost'):
    training_data = pd.DataFrame()
    need_deploy = False
    print("Starting Run")
    while self.data_flow != None and len(self.data_flow) > 0:
      
      #1. Data Ingestion
      next_data_flow = self.data_flow.pop()
      #2. Preprocessing
      print("Processing Data",next_data_flow['suffix'])
      new_data = process_data(next_data_flow['data'], next_data_flow['suffix'])
      
      #3 Drift Analysis Setup
      if self.check_drift(training_data,new_data) == False:
        print("✅ Sem drift. Dados serão ignorados para retreinamento.")
        continue

      training_data = pd.concat([training_data,new_data],axis=0)

      #4. Training
      print("Training Data after concat: ",next_data_flow['suffix'])
      training_results = train(training_data,model_name)
      
      #5. Evaluation
      print("Evaluating Data after concat: ",next_data_flow['suffix'])
      evaluation_results = evaluate(training_results)

      #6 Check If current Better than Last Best
      we_have_a_bew_best = self.check_if_new_winner(evaluation_results, training_results)
      need_deploy = we_have_a_bew_best or need_deploy
      if retrain==False:
        break

    #7 End Run
    if mlflow.active_run():
      mlflow.end_run() 

    #8 Deploy Flask App
    if need_deploy:
      self.update_production()
      update_production_model()
      
def setup_mlflow():
  client = MlflowClient()
  mlflow.set_tracking_uri("http://localhost:5000")
  mlflow.set_registry_uri("sqlite:///mlflow.db")
  if client.get_experiment_by_name(EXPERIMENT_NAME) == None:
    client.create_experiment(EXPERIMENT_NAME)
  return client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

def setup_data_flow():
  data1 = pd.read_csv('../data/data1.csv', encoding='latin-1')
  data2 = pd.read_csv('../data/data2.csv', encoding='latin-1')
  data_flow = [{"data":data1,"suffix":"1"},{"data":data2,"suffix":"2"}]
  return data_flow

if __name__ == "__main__":
  print("Waiting for mlflow:")
  mlflow_experiment_id = setup_mlflow()
  data_flow = setup_data_flow()

  pipeline = DiabetesDetectionPipeline(data_flow=data_flow)
  args = parser.parse_args()
  pipeline.run(True,args.model,mlflow_experiment_id)
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,roc_auc_score
import joblib
from sklearn.ensemble import RandomForestClassifier
import argparse
import mlflow
from dataclasses import dataclass
from imblearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


MAIN_USE_MLFLOW = False

@dataclass
class TrainingResults:
    model: object
    y_test: list
    y_pred: list
    name: str

@dataclass
class Metrics:
    f1_score: float
    accuracy: float
    precision: float
    recall: float
    roc_auc_score:float


def get_model_path(model_name):
   return f'../models/{model_name}_model.pkl'

def load_dataset(csv_path):
  data = pd.read_csv(csv_path, encoding='latin-1')
  return data

def plot_classes(data,title):
  data.value_counts().plot(kind='bar', color=['green','yellow', 'red'])
  plt.title(title)
  plt.xlabel('Class (0 = No Diabetes, 1 = Pre Diabetes, 2 - Diabetes)')
  plt.ylabel('Count')
  plt.show()



# Split 
def split_data(df):
  # Separate features and target
  X = df.drop('y', axis=1)  # Todas as colunas exceto 'y'
  y = df['y']  

  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  return X_train, X_test, y_train, y_test

def train_xgboost(data,model_name):
    # Train XGBoost Regression

    X_train,X_test, y_train, y_test = split_data(data)

    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    xgboost =  XGBClassifier(
    scale_pos_weight=scale_pos_weight,  # Valor calculado acima
    objective='binary:logistic',
    n_estimators=300,
    max_depth=4,            # Reduza para evitar overfitting
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
  )
    
    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('model', xgboost)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)


    return TrainingResults(model=xgboost, y_test=y_test, y_pred=y_pred, name=model_name)
   


def train_forest(data,model_name):

  X_train,X_test, y_train, y_test = split_data(data)

  rf =  RandomForestClassifier(
  n_estimators=200,
  max_depth=10,
  min_samples_split=5,
  random_state=42)

  pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', rf)
])

  # Train Random Forest
  pipeline.fit(X_train, y_train)

  # Evaluate on the test set
 # y_pred_rf = rf.predict(X_test)

  y_proba = pipeline.predict_proba(X_test)[:, 1]
  y_pred_rf = (y_proba >= 0.3).astype(int)  # Exemplo: threshold = 0.4
  
  return TrainingResults(model=rf, y_test=y_test, y_pred=y_pred_rf, name=model_name)
  
def save_model(model,model_name):
  joblib.dump(model,get_model_path(model_name))
  print("Model saved!")

def load_model(model_name):
  try:
    return joblib.load(get_model_path(model_name))
  except FileNotFoundError:
    return None

def calculate_metrics(y_test,y_pred):
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test,y_pred)
    return accuracy, precision, recall, f1, roc_auc

def handle_training_results(results:TrainingResults):
    accuracy, precision, recall,f1,roc_auc = calculate_metrics(results.y_test,results.y_pred)
    # Print classification report
    print(results.name + " Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Roc Auc Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(results.y_test, results.y_pred))
    return Metrics(f1_score=f1, accuracy=accuracy,precision=precision,recall=recall,roc_auc_score=roc_auc)

def train(data,model_name):
    
    if(model_name == "xgboost"):
      print("Training XGBOOST...")
      return train_xgboost(data,model_name)
    elif(model_name == "random_forest"):
      print("Training Random Forest...")
      return train_forest(data,model_name)
    else:
      print("Missing --model Parameter, Accepted Values: [","xgboost, ", "random_forest]")
      raise Exception("Error: No Model to Run")

   

if __name__ == "__main__":

   # Parse command-line arguments
  parser = argparse.ArgumentParser(description="Train a classification model.")
  parser.add_argument("--model", type=str, required=True, choices=["xgboost", "random_forest"],
                      help="Model to train: 'xgboost' or 'random_forest'")
  args = parser.parse_args()
  if MAIN_USE_MLFLOW:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_registry_uri("sqlite:///mlflow.db")
  data1 =load_dataset('../data/data_cleaned1.csv')
  data2 =load_dataset('../data/data_cleaned2.csv')


  if MAIN_USE_MLFLOW:
    with mlflow.start_run():
      print("Start")
      results = train(data1,args.model)
      handle_training_results(results)

      print("Train on New Data")
      new_data = pd.concat([data1,data2],axis=0)
      results = train(new_data,args.model)
      handle_training_results(results)
      save_model(results.model,results.name)
  else:
      print("Start")
      results = train(data2,args.model)
      handle_training_results(results)
      
      print("Train on New Data")
      new_data = pd.concat([data2,data1],axis=0)
      results = train(new_data,args.model)
      handle_training_results(results)
     

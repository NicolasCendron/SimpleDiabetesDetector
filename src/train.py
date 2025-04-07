from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
from sklearn.ensemble import RandomForestClassifier
import argparse
import mlflow
from dataclasses import dataclass


@dataclass
class TrainingResults:
    model: object
    y_test: list
    y_pred: list
    name: str


def get_model_path(model_name):
   return f'../models/{model_name}_model.pkl'

def load_dataset(csv_path):
  data = pd.read_csv(csv_path, encoding='latin-1')
  return data

def plot_classes(data,title):
  data.value_counts().plot(kind='bar', color=['blue', 'red'])
  plt.title(title)
  plt.xlabel('Class (0 = Ham, 1 = Spam)')
  plt.ylabel('Count')
  plt.show()

def convert_string_to_tfidf(X_train, X_test):
  # Convert text data to TF-IDF features
  print("Missing values in X_train:", X_train.isnull().sum())
  print("Missing values in X_test:", X_test.isnull().sum())
  X_train = X_train.fillna('')
  X_test = X_test.fillna('')
  vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 features for simplicity
  X_train_tfidf = vectorizer.fit_transform(X_train)
  X_test_tfidf = vectorizer.transform(X_test)
  print("TF-IDF transformation complete!")
  joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')
  print("TF-IDF vectorizer saved!")
  return X_train_tfidf,X_test_tfidf


# Balanceamento deixa para o Treinamento
def balance_data(data):
  print("Balance")
  # Separate features and target
  X = data['X']
  y = data['y']

  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  X_train_tfidf, x_test_tfidf = convert_string_to_tfidf(X_train,X_test)

  # Apply SMOTE to the training data
  smote = SMOTE(random_state=42)
  X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

  # Convert resampled data back to a DataFrame
  #plot_classes(y_train_resampled,'Balanced Class Distribution')
  return X_train_resampled, x_test_tfidf, y_train_resampled, y_test

def train_logistic(X_train,X_test_tfidf,y_train,y_test,model_name, old_model):
    # Train Logistic Regression

    if old_model != None:
       print("Continue Training where stopped")
       log_reg = old_model
    else:
       log_reg = LogisticRegression(random_state=42)
    
    
    log_reg.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = log_reg.predict(X_test_tfidf)
    
    
    return TrainingResults(model=log_reg, y_test=y_test, y_pred=y_pred, name=model_name)
   


def train_forest(X_train,X_test_tfidf,y_train,y_test,model_name,old_model):

  if old_model != None:
       print("Continue Training where stopped")
       rf = old_model
  else:
       rf =  RandomForestClassifier(  n_estimators=100,  # Number of trees
    max_depth=100,       # Limit tree depth
    min_samples_split=20,
    min_samples_leaf=20,
    random_state=42)

  # Train Random Forest
  rf.fit(X_train, y_train)

  # Evaluate on the test set
  y_pred_rf = rf.predict(X_test_tfidf,)
  # Adjust the threshold (e.g., 0.3)
  threshold = 0.3
  y_pred_adjusted = (y_pred_rf >= threshold).astype(int)
  
  return TrainingResults(model=rf, y_test=y_test, y_pred=y_pred_adjusted, name=model_name)
  
def save_model(model,model_name):
  joblib.dump(model,get_model_path(model_name))
  mlflow.sklearn.log_model(model, model_name)
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
    return accuracy, precision, recall, f1

def handle_training_results(results:TrainingResults):
    accuracy, precision, recall,f1 = calculate_metrics(results.y_test,results.y_pred)
    # Print classification report
    print(results.name + " Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    print(classification_report(results.y_test, results.y_pred))


   

def train(data,model_name,old_model=None):
    X_train_tfidf,X_test_tfidf, y_train, y_test = balance_data(data)
    if(model_name == "logistic_regression"):
      print("Training Logistic Regression...")
      return train_logistic(X_train_tfidf, X_test_tfidf, y_train, y_test,model_name,old_model)
    elif(model_name == "random_forest"):
      print("Training Random Forest...")
      return train_forest(X_train_tfidf, X_test_tfidf, y_train, y_test,model_name,old_model)
    else:
      print("Missing --model Parameter, Accepted Values: [","logistic_regression, ", "random_forest]")
      raise Exception("Error: No Model to Run")


if __name__ == "__main__":
   # Parse command-line arguments
  parser = argparse.ArgumentParser(description="Train a spam classification model.")
  parser.add_argument("--model", type=str, required=True, choices=["logistic_regression", "random_forest"],
                      help="Model to train: 'logistic_regression' or 'random_forest'")
  args = parser.parse_args()
  mlflow.set_tracking_uri("http://localhost:5000")
  mlflow.set_registry_uri("sqlite:///mlflow.db")
  data1 =load_dataset('../data/spam_cleaned1.csv')
  data2 =load_dataset('../data/spam_cleaned2.csv')
  with mlflow.start_run():
    print("Start")
    results = train(data2,args.model)
    handle_training_results(results)

    save_model(results.model,results.name)
    old_model = load_model(args.model)

    print("Train on New Data")
    results = train(data1,args.model,old_model)
    handle_training_results(results)
    save_model(results.model,results.name)

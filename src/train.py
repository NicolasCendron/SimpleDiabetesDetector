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



# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a spam classification model.")
parser.add_argument("--model", type=str, required=True, choices=["logistic_regression", "random_forest"],
                    help="Model to train: 'logistic_regression' or 'random_forest'")
args = parser.parse_args()

def load_dataset():
  data = pd.read_csv('../data/spam_cleaned_original.csv', encoding='latin-1')
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

def train_logistic(X_train,X_test_tfidf,y_train,y_test):
    # Train Logistic Regression
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = log_reg.predict(X_test_tfidf)
    handle_training_results(y_test,y_pred,"Logistic Regression")
    joblib.dump(log_reg, '../models/logistic_regression_model.pkl')
   
    mlflow.sklearn.log_model(log_reg, args.model)
    print("Model saved!")



def train_forest(X_train,X_test_tfidf,y_train,y_test):
  # Train Random Forest
  rf = RandomForestClassifier(  n_estimators=100,  # Number of trees
    max_depth=100,       # Limit tree depth
    min_samples_split=20,
    min_samples_leaf=20,
    random_state=42)
  rf.fit(X_train, y_train)

  # Evaluate on the test set
  y_pred_rf = rf.predict(X_test_tfidf,)
  # Adjust the threshold (e.g., 0.3)
  threshold = 0.3
  y_pred_adjusted = (y_pred_rf >= threshold).astype(int)
  # Calculate metrics
  handle_training_results(y_test,y_pred_adjusted,"Random Forest")
  joblib.dump(rf, '../models/random_forest_model.pkl')
  print("ANtes do ERRo")
  mlflow.sklearn.log_model(rf, args.model)
  print("Model saved!")

def calculate_metrics(y_test,y_pred):
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

def handle_training_results(y_test, y_pred,model_name):
    accuracy, precision, recall,f1 = calculate_metrics(y_test,y_pred)
    # Print classification report
    print(model_name + " Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    print(classification_report(y_test, y_pred))

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_registry_uri("sqlite:///mlflow.db")
with mlflow.start_run():

  data =load_dataset()
  X_train_tfidf,X_test_tfidf, y_train, y_test = balance_data(data)

  if( args.model == "logistic_regression"):
    print("Training Logistic Regression...")
    train_logistic(X_train_tfidf, X_test_tfidf, y_train, y_test)
  elif(args.model == "random_forest"):
    print("Training Random Forest...")
    train_forest(X_train_tfidf, X_test_tfidf, y_train, y_test)
  else:
    print("Missing --model Parameter, Accepted Values: [","logistic_regression, ", "random_forest]")
    raise Exception("Error: No Model to Run")
  #mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/{args.model}", args.model)

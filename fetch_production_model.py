import mlflow
import joblib
EXPERIMENT_NAME = 'Diabetes-Prediction'
mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Absolute path in Docker
mlflow.set_experiment(EXPERIMENT_NAME)
model = mlflow.pyfunc.load_model("models:/Diabetes-Model/Production")
joblib.dump(model, 'production_model.joblib')
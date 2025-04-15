from fastapi import FastAPI
import mlflow.models
import mlflow.models.model
from pydantic import BaseModel
import joblib
import uvicorn
import argparse
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = 'Diabetes-Model'
# Define the input schema
class PatientData(BaseModel):
    HighBP: float
    HighChol: float
    CholCheck: float
    BMI:float
    Smoker:float
    Stroke:float
    HeartDiseaseorAttack:float
    PhysActivity:float
    Fruits:float
    Veggies:float
    HvyAlcoholConsump:float
    AnyHealthcare:float
    NoDocbcCost:float
    GenHlth:float
    MentHlth:float
    PhysHlth:float
    DiffWalk:float
    Sex:float
    Age:float
    Education:float
    Income:float

# Initialize FastAPI app
app = FastAPI()

# Define the prediction endpoint
@app.post("/predict")
def predict(data: PatientData):
    input_data = data.dict()

    
    prediction = model.predict([list(input_data.values())])[0]
    
    # Log no MLflow
    # with mlflow.start_run():
    #     mlflow.log_params(input_data)
    #     mlflow.log_metric("prediction", prediction)
    
    if int(prediction) == 1:
        return {"prediction: Risk of diabetes detected. See your doctor."}
    else:
      return {"prediction: No diabetes risk detected, but keep taking care"}

@app.on_event("startup")
def startup():
    global model
    try:
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_registry_uri("sqlite:///mlflow.db")
            client = MlflowClient()
            registered_models = client.search_registered_models()
            for model in registered_models:
                print(f"Modelo: {model.name}")

            model_name = MODEL_NAME
            stage = "Production"
            model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        # Encerre a API se o modelo não puder ser carregado
        raise RuntimeError("Modelo não disponível.")

# Carregue o modelo em produção do MLflow
def update_production_model():

  uvicorn.run(app, host="0.0.0.0", port=8000)

# Run the app
if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)

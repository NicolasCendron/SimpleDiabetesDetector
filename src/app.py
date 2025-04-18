from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import mlflow
import pandas as pd

EXPERIMENT_NAME = 'Diabetes-Prediction'

# Define the input schema
class PatientData(BaseModel):
    HighBP: float
    HighChol : float
    CholCheck : float
    BMI:float
    Smoker :float
    Stroke :float
    HeartDiseaseorAttack :float
    PhysActivity :float
    Fruits:float
    Veggies :float
    HvyAlcoholConsump :float
    AnyHealthcare:float
    NoDocbcCost :float
    GenHlth :float
    MentHlth :float
    PhysHlth :float
    DiffWalk :float
    Sex:float
    Age :float
    Education:float
    Income :float

# Initialize FastAPI app
app = FastAPI()

# Define the prediction endpoint
@app.post("/predict")
def predict(data: PatientData):
    # Get column names from the model's input schema
    input_schema = model.metadata.get_input_schema()
    column_order = [col.name for col in input_schema.inputs]

    # Convert input data to a DataFrame in the correct order
    input_df = pd.DataFrame([data.dict()], columns=column_order)

    # Ensure data types match the schema (e.g., convert to float)
    input_df = input_df.astype({col: "float64" for col in column_order})

    # Make prediction
    prediction = model.predict(input_df)[0]
    return {"prediction": int(prediction)}
    
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
            mlflow.set_experiment(EXPERIMENT_NAME)
            model = mlflow.pyfunc.load_model("models:/Diabetes-Model/Production")
            
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        # Encerre a API se o modelo não puder ser carregado
        raise RuntimeError("Modelo não disponível.")

# Carregue o modelo em produção do MLflow
def deploy_production_model():
  uvicorn.run(app, host="0.0.0.0", port=8000)

# Run the app
if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)

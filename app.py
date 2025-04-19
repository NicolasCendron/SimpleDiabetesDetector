from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import mlflow
import pandas as pd
import logging
import traceback  # Add for error stack traces

EXPERIMENT_NAME = 'Diabetes-Prediction'
#MLFLOW_TRACKING_URL = "http://host.docker.internal:5000"
#MLFLOW_TRACKING_URL = "http://localhost:5000/"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Log to stdout (required for Docker)
)
logger = logging.getLogger(__name__)


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
    try:
        logger.info("Received prediction request: %s", data.dict())
        input_schema = model.metadata.get_input_schema()
        column_order = [col.name for col in input_schema.inputs]

        # Convert input data to a DataFrame in the correct order
        input_df = pd.DataFrame([data.dict()], columns=column_order)

        # Ensure data types match the schema (e.g., convert to float)
        input_df = input_df.astype({col: "float64" for col in column_order})

        # Make prediction
        prediction = model.predict(input_df)[0]

        if int(prediction) == 1:
            return {"prediction: Risk of diabetes detected. See your doctor."}
        else:
            return {"prediction: No diabetes risk detected, but keep taking care"}
    except:
        logger.error("PREDICTION ERROR: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    try:
        # Ping MLflow server
        client = mlflow.MlflowClient()
        experiments = client.search_experiments()
        
        return {"status": "healthy", "mlflow_connected": True, "#experiments":experiments.count()}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.on_event("startup")
def startup():
    global model
    try:
            mlflow.set_tracking_uri("sqlite:////app/mlflow.db")  # Absolute path in Docker
            #mlflow.set_registry_uri("mlflow.db")
            logger.info("Initializing MLflow connection...: http://host.docker.internal:5000")
            logger.info("Loading production model...")
            mlflow.set_experiment(EXPERIMENT_NAME)
            model = mlflow.pyfunc.load_model("models:/Diabetes-Model/Production")
            logger.info("Model loaded successfully")
            
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        logger.error("MODEL LOAD FAILED: %s", traceback.format_exc())
        # Encerre a API se o modelo não puder ser carregado
        raise RuntimeError("Modelo não disponível.")

# Run the app
#if __name__ == "__main__":
  #uvicorn.run(app, host="0.0.0.0", port=8000)

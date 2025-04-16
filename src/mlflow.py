from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from mlflow.models.signature import infer_signature

# 1. Configurar MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Iris-Classification")

# 2. Treinar e registrar modelo
with mlflow.start_run() as run:
    # Carregar dados
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Treinar modelo
    model = LogisticRegression()
    model.fit(X, y)
    
    # Logar métricas
    accuracy = model.score(X, y)
    mlflow.log_metric("accuracy", accuracy)
    
    # Logar modelo
    signature = infer_signature(X, model.predict(X))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature
    )
    
    # Registrar no Model Registry
    model_uri = f"runs:/{run.info.run_id}/model"
    registered_model = mlflow.register_model(model_uri, "Iris-Model")

# 3. Transicionar para Produção
client = MlflowClient()
client.transition_model_version_stage(
    name="Iris-Model",
    version=registered_model.version,
    stage="Production"
)

# 4. Criar API FastAPI
app = FastAPI()

# Carregar modelo em produção
model_prod = mlflow.pyfunc.load_model("models:/Iris-Model/Production")

@app.post("/predict")
def predict(features: list):
    return {"prediction": int(model_prod.predict([features])[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
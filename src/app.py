from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
import argparse






# Define the input schema
class TextInput(BaseModel):
    text: str

# Initialize FastAPI app
app = FastAPI()
MODEL = None
# Define the prediction endpoint
@app.post("/predict")
def predict(input: TextInput):

    # Make a prediction
    prediction = model.predict(text_tfidf)
    # Return the result
   
    result = "Not a Spam" if int(prediction[0]) == 0 else  "You found a Spam"
    print("Result is: ",result, int(prediction[0]))
    return {"prediction": result,"value":int(prediction[0])}


def deploy(model_name):
   

# Run the app
if __name__ == "__main__":
    # Parse command-line arguments
  parser = argparse.ArgumentParser(description="Train a spam classification model.")
  parser.add_argument("--model", type=str, required=True, choices=["logistic_regression", "random_forest"],
                      help="Model to train: 'logistic_regression' or 'random_forest'")
  args = parser.parse_args()
  XG_BOOST = joblib.load('../models/xgboost_model.pkl')
  RANDOM_FOREST = joblib.load('../models/random_forest_model.pkl')
  print("Missing --model Parameter, Accepted Values: [","logistic_regression, ", "random_forest]")
  uvicorn.run(app, host="0.0.0.0", port=8000)
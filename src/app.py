from fastapi import FastAPI
from pydantic import BaseModel
import joblib

import argparse


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a spam classification model.")
parser.add_argument("--model", type=str, required=True, choices=["logistic_regression", "random_forest"],
                    help="Model to train: 'logistic_regression' or 'random_forest'")
args = parser.parse_args()

if( args.model == "logistic_regression"):
  print("Loading Logistic Regression API...")
  model = joblib.load('../models/logistic_regression_model.pkl')
elif(args.model == "random_forest"):
  print("Loading Random Forest API...")
  model = joblib.load('../models/random_forest_model.pkl')
else:
   print("Missing --model Parameter, Accepted Values: [","logistic_regression, ", "random_forest]")

# Load the vectorizer
vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')  # Save the vectorizer during training

# Define the input schema
class TextInput(BaseModel):
    text: str

# Initialize FastAPI app
app = FastAPI()

# Define the prediction endpoint
@app.post("/predict")
def predict(input: TextInput):
    # Preprocess the input text
    text_tfidf = vectorizer.transform([input.text])
    # Make a prediction
    prediction = model.predict(text_tfidf)
    # Return the result
   
    result = "Not a Spam" if int(prediction[0]) == 0 else  "You found a Spam"
    print("Result is: ",result, int(prediction[0]))
    return {"prediction": result,"value":int(prediction[0])}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
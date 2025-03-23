from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the model
model = joblib.load('../models/logistic_regression_model.pkl')

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
    return {"prediction": result}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
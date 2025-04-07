# Spam Classification Project

This project aims to classify SMS messages as spam or ham using machine learning.

## Before Start

Download SPAM dataset: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Install Python 3.10 (recommended virtual env)

## How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

1.1 Split Data between "Original" and "New" to Monitor Drift Later

python split_csv.py

2. Preprocess data:

   python preprocess.py

3. Train the model (on "Original" data):

   python train.py --model logistic_regression

   python train.py --model random_forest

4. Deploy the model (Locally):

   python app.py --model logistic_regression

   python app.py --model random_forest

   Once Running, Test using POST method to http://localhost:8000/predict

   Body Examples:
   SPAM
   {
   "text":"Gimme credit card info to win big money"
   }
   NOT SPAM:
   {
   "text":"Son, remember to wear a Coat. "
   }

5. Monitor Drift

   python monitor_drift.py

Run mlflow
mlflow server \
 --backend-store-uri sqlite:///mlflow.db \
 --default-artifact-root ./mlruns \
 --host 0.0.0.0 \
 --port 5000

License
This project is licensed under the MIT License.

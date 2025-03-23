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

2. Preprocess data:

   python preprocess.py

3. Train the model:

   python train.py (Edit the LOGISTIC variable to cycle between Regression and Forest Ensemble)

   TODO: change to a parameter

4. Deploy the model (Locally):

   python app.py

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

License
This project is licensed under the MIT License.

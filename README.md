# Diabetes Detection Project

This project aims to predict diabetes on patients given medical featurs.

## Before Start

Download Diabetes dataset: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

Install Python 3.10 (recommended virtual env)

## How to Run

### 0. Run mlflow

mlflow server \
 --default-artifact-root ./mlruns \
 --host 0.0.0.0 \
 --port 5000

mlflow server \
 --default-artifact-root ./mlruns \
 --backend-store-uri sqlite:///mlflow.db \
 --host 0.0.0.0 \
 --port 5000

### 1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### 2. Split Data between "Original" and "New" to Monitor Drift Later

python split_csv.py

### 3. Run pipeline:

python pipeline.py --model xgboost

or

python pipeline.py --model random_forest

### 4. Check Experiments and Models on MLFlow (http://localhost:5000/)

### 5. Fetch Production Model

python fetch_production_model.py

### 6. Run Dockerfile:

docker build -t diabetes-api .
docker run -p 8000:8000 --log-driver=local --name diabetes-api diabetes-api

(Keep in mind your mlflow should be running)

### 7. Run predictions on http://localhost:8000/predict (POST)

For Documentation on features refer to Kaggle page.

### Body Examples

### Expected: High Diabetes Risk

```json
{
  "HighBP": 1, // Hypertension (major risk factor)
  "HighChol": 1, // High cholesterol
  "CholCheck": 0, // No cholesterol check in the last 5 years
  "BMI": 40, // Obesity (BMI ≥ 30 is a key diabetes risk)
  "Smoker": 1, // Smoking history
  "Stroke": 1, // Previous stroke
  "HeartDiseaseorAttack": 1, // Heart disease history
  "PhysActivity": 0, // No physical activity
  "Fruits": 0, // Rarely/never eats fruits
  "Veggies": 0, // Rarely/never eats vegetables
  "HvyAlcoholConsump": 0, // No heavy drinking (paradoxically, moderate alcohol may reduce risk)
  "AnyHealthcare": 0, // No access to healthcare
  "NoDocbcCost": 1, // Couldn’t see a doctor due to cost
  "GenHlth": 5, // Poor general health (scale: 1=excellent, 5=poor)
  "MentHlth": 30, // Frequent mental distress (days/month)
  "PhysHlth": 30, // Frequent physical health issues (days/month)
  "DiffWalk": 1, // Difficulty walking/climbing stairs
  "Sex": 1, // Male (higher diabetes risk)
  "Age": 80, // Older age (risk increases with age)
  "Education": 1, // Low education level (high school or less)
  "Income": 1 // Low income (<$10k/year)
}
```

### Expected: Low Diabetes Risk

```json
{
  "HighBP": 0, // No hypertension
  "HighChol": 0, // Normal cholesterol
  "CholCheck": 1, // Cholesterol checked in the last 5 years
  "BMI": 22, // Healthy weight (BMI 18.5–24.9)
  "Smoker": 0, // Never smoked
  "Stroke": 0, // No history of stroke
  "HeartDiseaseorAttack": 0, // No heart disease
  "PhysActivity": 1, // Regular physical activity
  "Fruits": 1, // Daily fruit consumption
  "Veggies": 1, // Daily vegetable consumption
  "HvyAlcoholConsump": 0, // No heavy drinking
  "AnyHealthcare": 1, // Access to healthcare
  "NoDocbcCost": 0, // No cost barriers to seeing a doctor
  "GenHlth": 1, // Excellent general health (1=excellent)
  "MentHlth": 0, // No poor mental health days
  "PhysHlth": 0, // No poor physical health days
  "DiffWalk": 0, // No difficulty walking
  "Sex": 0, // Female (lower risk statistically)
  "Age": 30, // Younger age (lower risk)
  "Education": 6, // Higher education (e.g., college degree)
  "Income": 8 // Higher income (>$75k/year)
}
```

License
This project is licensed under the MIT License.

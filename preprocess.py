import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder


FIX_BMI = False
FIX_AGE = False

def check_data(data):
   # Inspect the first few rows
  print("Head", data.head())

  # Check for missing values
  print("Missing Values", data.isnull().sum())

  #plotClasses(data,'Original Class Distribution')

def save_data(data,suffix):
    data.to_csv('data/data_cleaned' + suffix +'.csv', index=False)
    print("Data saves saved!")
   


def plotClasses(data,title):
  data['y'].value_counts().plot(kind='bar', color=['blue', 'red'])
  plt.title(title)
  plt.xlabel('Class (0 = No Diabetes, 1 = Pre-Diabetes, 2 = Diabetes)')
  plt.ylabel('Count')
  plt.show()


def fix_age(df):
  df = df[(df['Age'] >= 0) & (df['Age'] <= 120)]  # Filtrar idades realistas
  return df


def fix_BMI(df):
  df['BMI_Age'] = df['BMI'] * df['Age']  
  df['GenHlth_BMI'] = df['GenHlth'] * df['BMI']
  df = df.drop("BMI", axis=1)
  df = df.drop("Age", axis=1)
  df = df.drop("GenHlth", axis=1)
  return df


def fix_features(df):
  if FIX_AGE:
    df = fix_age(df)
  if FIX_BMI:
    df = fix_BMI(df)
  return df
  
def remove_prediabetes(df):
  df_filtered = df[df['y'] != 1]  # Remove pré-diabetes (categoria 1)
  df_filtered['y'] = df_filtered['y'].map({0: 0, 2: 1})  # Re-mapeia 2 para 1
  return df_filtered


def process_data(df,suffix=''):
  print("Process Data")
  df.drop_duplicates(inplace=True)
  df = df.rename(columns={"Diabetes_012": "y"})
  df = fix_features(df)
  df = remove_prediabetes(df)
  #check_data(df)
  save_data(df,suffix)
  return df

if __name__ == "__main__":
  data = pd.read_csv('data/data1.csv', encoding='latin-1')
  clean_data = process_data(data,'1')
 #plotClasses(clean_data,"Class Distribution 1")

  data2 = pd.read_csv('data/data2.csv', encoding='latin-1')
  clean_data2 = process_data(data2,'2')
  #plotClasses(clean_data,"Class Distribution 2")
  
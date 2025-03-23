import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"

def load_dataset():
  data = pd.read_csv('../data/spam.csv', encoding='latin-1')
  return data

def check_data(data):
   # Inspect the first few rows
  print(data.head())

  # Check for missing values
  print(data.isnull().sum())

  # Check the distribution of labels
  print(data['y'].value_counts())

  plotClasses(data,'Original Class Distribution')


def fix_classes(data):

  df1 = data[['v1','v2']]
  df1 = df1.rename(columns={"v1": "y", "v2": "X"})
  df1 = df1.dropna(subset=['X'])
  df1 = df1.fillna('')
  return df1

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text_function(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords and stem the tokens
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    # Join tokens back into a string
    return ' '.join(tokens)

def preprocess_text(data):
    # Apply preprocessing to the SMS column
    data['X'] = data['X'].apply(preprocess_text_function)

    # Encode labels (ham = 0, spam = 1)
    data['y'] = data['y'].map({'ham': 0, 'spam': 1})
    print("Data cleaned!")

    
    return data

def save_data(data):
    data.to_csv('../data/spam_cleaned.csv', index=False)
    print("Data saves saved!")
   


def plotClasses(data,title):
  data['y'].value_counts().plot(kind='bar', color=['blue', 'red'])
  plt.title(title)
  plt.xlabel('Class (0 = Ham, 1 = Spam)')
  plt.ylabel('Count')
  plt.show()


data = load_dataset()
fixed_data = fix_classes(data)

#Processed texts
processed_data = preprocess_text(fixed_data)
check_data(processed_data)
 # Save the cleaned data
save_data(processed_data)
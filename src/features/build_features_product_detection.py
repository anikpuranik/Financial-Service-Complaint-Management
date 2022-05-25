# Importing Libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Loading Data
def load_data(path):
    nltk.download('stopwords')
    nltk.download("punkt")
    dataset = pd.read_csv(path, usecols=["Product", "Consumer complaint narrative", "Date received"])
    data = dataset[dataset["Date received"] >= "2021-12-01"]
    data = data[data['Date received'] < "2022-01-01"]
    data = data[["Product", "Consumer complaint narrative"]]
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

# Text Prepocessing
def clean_text(data, column):
    stop_words = stopwords.words("english")
    stop_words.append("xxxx")
    
    processed_text = []
    for text in data[column]:
        text = " ".join([word for word in word_tokenize(text.lower())
                         if (word not in stop_words and word.isalpha())
                         ])
        processed_text.append(text)
    data[column] = processed_text
    return data

def run():
    data_load_path = "K://2022Python//Consumer Complaints//data//raw.csv"
    data_save_path = "K://2022Python//Consumer Complaints//data//processed_product_detection_data.csv"
    feature_column = "Consumer complaint narrative"
    target_column = "Product"

    print("Loading Data...")
    data = load_data(data_load_path)
    print("Cleaning Data...")
    data = clean_text(data, feature_column)
    print("Saving processed file...")
    data.to_csv(data_save_path, index=False)

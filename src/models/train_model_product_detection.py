# Importing Libraries
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Loading Data
def load_data(path):
    data = pd.read_csv(path)
    return data

# Embedding
def embedding_data(feature):
  tfidf = TfidfVectorizer(sublinear_tf=True, min_df=10, ngram_range=(1, 1), lowercase=False)
  features = tfidf.fit_transform(feature).toarray()
  return tfidf, features

# Training Model
def training_model(features, labels):
  model = LinearSVC()
  model.fit(features, labels)
  return model

def run():
    data_save_path = "K://2022Python//Consumer Complaints//data//processed_product_detection_data.csv"
    model_save_path = "K://2022Python//Consumer Complaints//models//product_detection_classifier.pkl"
    vectorizer_save_path = "K://2022Python//Consumer Complaints//models//text_vectorizer.pkl"

    print("Loading Data...")
    data = load_data(data_save_path)

    # Splitting features and labels
    Target_Column = "Product"
    labels = data[Target_Column]
    data.drop(columns=[Target_Column], inplace=True)
    feature_column  = data.columns[0]

    print("Performing Embedding...")
    vectorizer, features = embedding_data(data[feature_column])

    print("Generating Pickle For vectorizer")
    pickle.dump(vectorizer, open(vectorizer_save_path,"wb"))
    del vectorizer

    print("Training Model")
    SVM_Model = training_model(features, labels)

    print("Generating Pickle For model")
    pickle.dump(SVM_Model, open(model_save_path,"wb"))


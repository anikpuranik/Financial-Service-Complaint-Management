# Importing Libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Loading Dataset
def load_data(path):
    data = pd.read_csv(path)
    return data

# Training Model
def train_model(features, labels):
  model = DecisionTreeClassifier()
  model.fit(features, labels)
  return model

def run():
    data_load_path = "K://2022Python//Consumer Complaints//data//processed_consumer_disputed_data.csv"
    model_save_path = "K://2022Python//Consumer Complaints//models//consumer_disputed_classifier.pkl"
    
    print("Loading Data...")
    data = load_data(data_load_path)
    
    # Splitting features and labels
    Target_Column = "Consumer disputed?"
    labels = data[Target_Column]
    features = data.drop(columns=[Target_Column])
    
    print("Training Model")
    model = train_model(features, labels)
    print("Saving Model...")
    pickle.dump(model, open(model_save_path, "wb"))

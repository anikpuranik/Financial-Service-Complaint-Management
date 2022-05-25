# Importing Libraries
import numpy as np
import pandas as pd

# Loading Dataset
def load_data(path):
    data = pd.read_csv(path, usecols=["Product", "Issue", "State", "Company response to consumer", "Submitted via", "Timely response?", "Consumer disputed?"])
    return data

def product_dropped(data, products_dropped):
    for product in products_dropped:
        data.drop(data[data["Product"]==product].index, inplace=True)
    return data

def factorizing_high_data(data, columns):
    for column in columns:
        for value in data[column].value_counts().index[:10]:
            varname = "{}_{}".format(column, value)
            data[varname] = np.where(data[column]==value, 1, 0)
        data.drop(columns=column, inplace=True)
    return data

def factorizing_low_data(data, columns):
    for column in columns:
        temp = pd.get_dummies(data[column], drop_first=True, prefix=column)
        data = pd.concat([temp, data], axis=1)
        data.drop(columns=column, inplace=True)
    return data
    
def pre_processing(data, target_column, products_dropped, 
                   high_data_columns, low_data_columns):
    indices = data[target_column].dropna().index
    data = data.loc[indices].reset_index(drop=True)
    
    # removing outdated products
    data = product_dropped(data, products_dropped)
    
    # factorizing categorical features
    data = factorizing_high_data(data, high_data_columns)
    data = factorizing_low_data(data, low_data_columns)
    
    return data

def balancing_data(data):
    train_size = int(len(data[data["Consumer disputed?"]=="Yes"])*0.8)
    temp_yes_data = data[data["Consumer disputed?"]=="Yes"].sample(train_size)
    temp_no_data = data[data["Consumer disputed?"]=="No"].sample(2*len(temp_yes_data))
    training_data = pd.concat([temp_yes_data, temp_no_data], axis=0)
    return training_data

def run():
    data_load_path = "K://2022Python//Consumer Complaints//data//raw.csv"
    data_save_path = "K://2022Python//Consumer Complaints//data//processed_consumer_disputed_data.csv"
    Target_Column = "Consumer disputed?"
    
    products_dropped = ["Consumer Loan", "Bank account or service", "Credit reporting", "Credit card", 
                    "Other financial service", "Money transfers", "Payday loan", "Prepaid card", "Virtual currency"]
    high_data_columns = ["State", "Issue"]
    low_data_columns = ["Product", "Submitted via", "Company response to consumer", "Timely response?"]
    
    print("Loading Data...")
    data = load_data(data_load_path)
    print("Processing Data...")
    data = pre_processing(data, Target_Column, products_dropped, high_data_columns, low_data_columns)
    data = balancing_data(data)
    print("Saving processed file...")
    data.to_csv(data_save_path, index=False)

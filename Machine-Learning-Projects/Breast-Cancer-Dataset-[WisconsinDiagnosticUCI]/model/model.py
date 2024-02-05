import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def Logistic_Regression()


def main(_):
    data = pd.read_csv(r"../data/model_training_data.csv")
    
    x = data.drop(['target'],axis=1)
    y = data.target.to_numpy()
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=60)
    
    



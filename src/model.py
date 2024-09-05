import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the parameter grid for hyperparameter tuning
    params = {'max_depth' : [8,10,12,15], 'learning_rate' : [0.01,0.05,0.1,0.2], 'scale_pos_weight' : [2,3,4,5]}

    
    # Create and train the model with hyperparameter tuning
    model = XGBClassifier(random_state=42)
    grid_search = GridSearchCV(model, params, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    return best_model

def save_model(model, filename='titanic_model.joblib'):
    joblib.dump(model, filename)

def load_model(filename='titanic_model.joblib'):
    return joblib.load(filename)

def predict_survival(model, features):
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    return prediction[0], probability[0][1]

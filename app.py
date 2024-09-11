import streamlit as st
import pandas as pd
import numpy as np
from src.preprocess import load_data, preprocess_data
from src.model import train_model, save_model, load_model, predict_survival
import os

# Load and preprocess data
@st.cache_data
def get_data():
    df = load_data('data/titanic.csv')
    X, y = preprocess_data(df)
    return X, y

# Train or load model
@st.cache_resource
def get_model(X, y):
    model_path = 'titanic_model.joblib'
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = train_model(X, y)
        save_model(model, model_path)
    return model

# Streamlit app
def main():
    st.title('Titanic Survival Prediction')
    
    # Get data and model
    X, y = get_data()
    model = get_model(X, y)
    
    # Input fields
    st.sidebar.header('Enter Passenger Information')
    pclass = st.sidebar.selectbox('Passenger Class', [1, 2, 3])
    sex = st.sidebar.selectbox('Sex', ['male', 'female'])
    age = st.sidebar.slider('Age', 0, 100, 30)
    sibsp = st.sidebar.slider('Number of Siblings/Spouses Aboard', 0, 8, 0)
    parch = st.sidebar.slider('Number of Parents/Children Aboard', 0, 6, 0)
    fare = st.sidebar.number_input('Fare', min_value=0.0, max_value=600.0, value=32.2)
    
    # Encode sex
    sex_encoded = 1 if sex == 'female' else 0
    
    # Create feature array
    features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare]])
    
    # Make prediction
    if st.button('Predict Survival'):
        prediction, probability = predict_survival(model, features)
        
        st.subheader('Prediction Result')
        if prediction == 1:
            st.success(f'This passenger would have survived with a probability of {probability:.2%}')
        else:
            st.error(f'This passenger would not have survived with a probability of {(1-probability):.2%}')

if __name__ == '__main__':
    main()
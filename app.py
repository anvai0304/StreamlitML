# import streamlit as st
# import pandas as pd
# import numpy as np
# from src.preprocess import load_data, preprocess_data
# from src.model import train_model, save_model, load_model, predict_survival
# import os

# # Set page config
# st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

# # Custom CSS for better contrast, styling, and alignment
# st.markdown("""
# <style>
#     .stApp {
#         background-color: #f0f8ff;
#     }
#     .main .block-container {
#         padding-top: 2rem;
#         max-width: 800px;
#         margin: 0 auto;
#     }
#     h1, h2, h3, .stButton>button, label, .stSelectbox>div>label, .stSlider>div>label {
#         color: #2c3e50;
#     }
#     .stTextInput>div>div>input, .stSelectbox>div>div>select {
#         background-color: white;
#         color: #2c3e50;
#     }
#     .stButton>button {
#         background-color: #3498db;
#         color: white;
#         display: block;
#         margin: 0 auto;
#     }
#     .stButton>button:hover {
#         background-color: #2980b9;
#     }
#     [data-testid="stHorizontalBlock"] > div > div {
#         border: 1px solid #e0e0e0;
#         border-radius: 5px;
#         padding: 10px;
#         margin: 5px;
#         background-color: #ffffff;
#     }
#     .stRadio > label {
#         color: #2c3e50;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Load and preprocess data
# @st.cache_data
# def get_data():
#     df = load_data('data/titanic.csv')
#     X, y = preprocess_data(df)
#     return X, y

# # Train or load model
# @st.cache_resource
# def get_model(X, y):
#     model_path = 'titanic_model.joblib'
#     if os.path.exists(model_path):
#         model = load_model(model_path)
#     else:
#         model = train_model(X, y)
#         save_model(model, model_path)
#     return model

# # Streamlit app
# def main():
#     st.title('🚢 Titanic Survival Prediction')
    
#     # Get data and model
#     X, y = get_data()
#     model = get_model(X, y)
    
#     st.header("Enter Passenger Information")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("Personal Details")
#         pclass = st.selectbox('Passenger Class', [1, 2, 3], format_func=lambda x: f"{x} (1st, 2nd, or 3rd)")
#         sex = st.radio('Sex', ['male', 'female'])
#         age = st.slider('Age', 0, 100, 30)
    
#     with col2:
#         st.subheader("Travel Information")
#         sibsp = st.slider('Number of Siblings/Spouses Aboard', 0, 8, 0)
#         parch = st.slider('Number of Parents/Children Aboard', 0, 6, 0)
#         fare = st.number_input('Fare (in £)', min_value=0.0, max_value=600.0, value=32.2, step=0.1)
    
#     # Encode sex
#     sex_encoded = 1 if sex == 'female' else 0
    
#     # Create feature array
#     features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare]])
    
#     # Make prediction
#     if st.button('Predict Survival'):
#         prediction, probability = predict_survival(model, features)
        
#         st.subheader('Prediction Result')
#         if prediction == 1:
#             st.success(f'This passenger would likely have survived with a probability of {probability:.2%}')
#         else:
#             st.error(f'This passenger would likely not have survived with a probability of {(1-probability):.2%}')

# if __name__ == '__main__':
#     main()

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
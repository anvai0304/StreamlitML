# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from src.preprocess import load_data, preprocess_data
# from src.model import train_model, save_model, load_model, predict_survival
# import os

# # Set page config
# st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

# # Custom CSS (same as before, with additional styling)
# st.markdown("""
# <style>
#     .stApp {
#         background-color: #f0f8ff;
#     }
#     .main .block-container {
#         padding-top: 2rem;
#         max-width: 1200px;
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
#     .prediction-box {
#         padding: 20px;
#         border-radius: 10px;
#         margin-top: 20px;
#     }
#     .survivor {
#         background-color: #d4edda;
#         border: 1px solid #c3e6cb;
#     }
#     .non-survivor {
#         background-color: #f8d7da;
#         border: 1px solid #f5c6cb;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Load and preprocess data
# @st.cache_data
# def get_data():
#     df = load_data('data/titanic.csv')
#     X, y = preprocess_data(df)
#     return df, X, y

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

# # Create visualizations
# @st.cache_data
# def create_visualizations(df):
#     # Survival rate by passenger class
#     fig1, ax1 = plt.subplots(figsize=(10, 6))
#     sns.barplot(x='Pclass', y='Survived', data=df, ax=ax1)
#     ax1.set_title('Survival Rate by Passenger Class')
#     ax1.set_xlabel('Passenger Class')
#     ax1.set_ylabel('Survival Rate')

#     # Age distribution
#     fig2, ax2 = plt.subplots(figsize=(10, 6))
#     sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', ax=ax2)
#     ax2.set_title('Age Distribution of Survivors and Non-Survivors')
#     ax2.set_xlabel('Age')
#     ax2.set_ylabel('Count')

#     return fig1, fig2

# # Streamlit app
# def main():
#     st.title('🚢 Titanic Survival Predictor')
    
#     # Get data and model
#     df, X, y = get_data()
#     model = get_model(X, y)
    
#     # Create tabs
#     tab1, tab2, tab3 = st.tabs(["Prediction", "Data Insights", "Historical Context"])
    
#     with tab1:
#         st.header("Enter Passenger Information")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("Personal Details")
#             pclass = st.selectbox('Passenger Class', [1, 2, 3], format_func=lambda x: f"{x} (1st, 2nd, or 3rd)")
#             sex = st.radio('Sex', ['male', 'female'])
#             age = st.slider('Age', 0, 100, 30)
        
#         with col2:
#             st.subheader("Travel Information")
#             sibsp = st.slider('Number of Siblings/Spouses Aboard', 0, 8, 0)
#             parch = st.slider('Number of Parents/Children Aboard', 0, 6, 0)
#             fare = st.number_input('Fare (in £)', min_value=0.0, max_value=600.0, value=32.2, step=0.1)
        
#         # Encode sex
#         sex_encoded = 1 if sex == 'female' else 0
        
#         # Create feature array
#         features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare]])
        
#         # Make prediction
#         if st.button('Predict Survival'):
#             prediction, probability = predict_survival(model, features)
            
#             st.subheader('Prediction Result')
#             if prediction == 1:
#                 st.markdown(f'<div class="prediction-box survivor"><h3>Likely Survivor</h3><p>This passenger would likely have survived with a probability of {probability:.2%}</p></div>', unsafe_allow_html=True)
#             else:
#                 st.markdown(f'<div class="prediction-box non-survivor"><h3>Likely Non-Survivor</h3><p>This passenger would likely not have survived with a probability of {(1-probability):.2%}</p></div>', unsafe_allow_html=True)
    
#     with tab2:
#         st.header("Data Insights")
#         fig1, fig2 = create_visualizations(df)
#         st.pyplot(fig1)
#         st.pyplot(fig2)
        
#         st.subheader("Key Statistics")
#         col1, col2, col3 = st.columns(3)
#         col1.metric("Total Passengers", len(df))
#         col2.metric("Survival Rate", f"{df['Survived'].mean():.2%}")
#         col3.metric("Average Fare", f"£{df['Fare'].mean():.2f}")
    
#     with tab3:
#         st.header("Historical Context")
#         st.subheader("The Titanic Disaster")
#         st.write("""
#         The RMS Titanic sank in the early morning hours of April 15, 1912, after striking an iceberg during her maiden voyage from Southampton to New York City. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making it one of the deadliest peacetime maritime disasters in modern history.

#         The sinking of the Titanic shocked the world and led to major improvements in maritime safety. It remains one of the most famous shipwrecks in history, inspiring numerous books, articles, and films.
#         """)
        
#         st.subheader("Factors Affecting Survival")
#         st.write("""
#         Several factors influenced a passenger's likelihood of survival:
#         - Class: First-class passengers had a higher survival rate.
#         - Gender: Women and children were prioritized for lifeboats.
#         - Age: Young children had a higher survival rate.
#         - Location on the ship: Passengers closer to lifeboats had better chances.

#         Our prediction model takes some of these factors into account to estimate survival probability.
#         """)

# if __name__ == '__main__':
    # main()

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
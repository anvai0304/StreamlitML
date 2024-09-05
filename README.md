# Titanic Survival Prediction

This project uses machine learning to predict the survival of Titanic passengers based on various features.

## Setup

1. Clone this repository.
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Place your Titanic dataset (CSV file) in the `data/` directory.

## Running the Application

To run the Streamlit app:

```
streamlit run app.py
```

This will start the app and open it in your default web browser.

## Project Structure

- `data/`: Contains the Titanic dataset.
- `notebooks/`: Contains the Jupyter notebook with the original model development.
- `src/`: Contains the source code for data preprocessing and model training.
- `app.py`: The main Streamlit application.

## Model

The model uses XGBoost for classification. It's trained on features including passenger class, sex, age, number of siblings/spouses aboard, number of parents/children aboard, and fare.
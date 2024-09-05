import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    return pd.read_csv('model/train (1).csv')

def preprocess_data(df):
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X = df[features]
    y = df['Survived']
    
    le = LabelEncoder()
    X['Sex'] = le.fit_transform(X['Sex'])
    
    X['Age'].fillna(X['Age'].median(), inplace=True)
    X['Fare'].fillna(X['Fare'].median(), inplace=True)
    
    return X, y
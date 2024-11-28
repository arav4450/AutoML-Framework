import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    # Fill missing values
    imputer = SimpleImputer(strategy='mean')
    df.iloc[:, :] = imputer.fit_transform(df)

    # Scale numerical features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))

    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df.select_dtypes(include=['object']))

    # Combine and return
    return pd.concat([pd.DataFrame(scaled_features), pd.DataFrame(encoded_features)], axis=1)


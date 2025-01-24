import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from category_encoders.woe import WOEEncoder
from typing import Optional

class FeatureEngineering:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with the dataset.
        :param data: Input DataFrame.
        """
        self.data = data.copy()
        print(self.data.head())

    def create_aggregate_features(self):
        """
        Create aggregate features such as total, average, count, and standard deviation of transaction amounts.
        """
        self.data['TotalTransactionAmount'] = self.data.groupby('CustomerId')['Amount'].transform('sum')
        self.data['AvgTransactionAmount'] = self.data.groupby('CustomerId')['Amount'].transform('mean')
        self.data['TransactionCount'] = self.data.groupby('CustomerId')['Amount'].transform('count')
        self.data['StdTransactionAmount'] = self.data.groupby('CustomerId')['Amount'].transform('std')
        print(self.data.head())
    def extract_datetime_features(self, datetime_column: str):
        """
        Extract features such as hour, day, month, and year from a datetime column.
        :param datetime_column: Name of the datetime column.
        """
        self.data[datetime_column] = pd.to_datetime(self.data[datetime_column])
        self.data['TransactionHour'] = self.data[datetime_column].dt.hour
        self.data['TransactionDay'] = self.data[datetime_column].dt.day
        self.data['TransactionMonth'] = self.data[datetime_column].dt.month
        self.data['TransactionYear'] = self.data[datetime_column].dt.year
        return self.data.head()
    def encode_categorical_variables(self, columns: list, method: str = "onehot"):
        """
        Encode categorical variables using OneHot or Label Encoding.
        :param columns: List of categorical columns to encode.
        :param method: Encoding method ("onehot" or "label").
        """
        if method == "onehot":
            encoder = OneHotEncoder(sparse_output=False, drop="first")
            encoded = pd.DataFrame(encoder.fit_transform(self.data[columns]), columns=encoder.get_feature_names_out(columns))
            self.data = pd.concat([self.data.drop(columns, axis=1), encoded], axis=1)
        elif method == "label":
            for col in columns:
                encoder = LabelEncoder()
                self.data[col] = encoder.fit_transform(self.data[col])
        else:
            raise ValueError("Invalid encoding method. Choose 'onehot' or 'label'.")
        return self.data.head()
    def handle_missing_values(self, method: str = "impute", strategy: str = "mean", columns: list = None):
        """
        Handle missing values in the dataset using imputation or removal.

        :param method: str - "impute" or "remove".
        :param strategy: str - Strategy for imputation ("mean", "median", "mode"). Only used if method="impute".
        :param columns: list - Specific columns to handle. If None, applies to all columns.
        :raises ValueError: If the method or strategy is invalid.
        """
        if self.data is None or self.data.empty:
            raise ValueError("The dataset is empty or not loaded.")

        # Apply to all columns if no specific columns are provided
        if columns is None:
            columns = self.data.columns

        # Handle missing values
        if method == "impute":
            if strategy not in ["mean", "median", "mode"]:
                raise ValueError("Invalid strategy. Choose 'mean', 'median', or 'mode'.")
            for col in columns:
                if self.data[col].isnull().any():
                    if strategy == "mean":
                        self.data[col].fillna(self.data[col].mean(), inplace=True)
                    elif strategy == "median":
                        self.data[col].fillna(self.data[col].median(), inplace=True)
                    elif strategy == "mode":
                        self.data[col].fillna(self.data[col].mode().iloc[0], inplace=True)
        elif method == "remove":
            self.data.dropna(subset=columns, inplace=True)
        else:
            raise ValueError("Invalid method. Choose 'impute' or 'remove'.")


    def normalize_numerical_features(self, columns: list, method: str = "minmax"):
        """
        Normalize numerical features using Min-Max or Standard Scaling.
        :param columns: List of numerical columns to scale.
        :param method: Scaling method ("minmax" or "standard").
        """
        if method == "minmax":
            scaler = MinMaxScaler()
        elif method == "standard":
            scaler = StandardScaler()
        else:
            raise ValueError("Invalid method. Choose 'minmax' or 'standard'.")

        self.data[columns] = scaler.fit_transform(self.data[columns])

    def calculate_woe_iv(self, target_column: str, columns: list):
        """
        Calculate Weight of Evidence (WOE) and Information Value (IV) for categorical variables.
        :param target_column: Target column for WOE/IV calculation.
        :param columns: List of columns to calculate WOE/IV.
        """
        encoder = WOEEncoder()
        self.data[columns] = encoder.fit_transform(self.data[columns], self.data[target_column])

    def get_data(self) -> pd.DataFrame:
        """
        Return the transformed dataset.
        """
        return self.data

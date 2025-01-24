# src/eda_processor.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")  # Set seaborn style for consistency


class EDAProcessor:
    def __init__(self, filepath: str):
        """Initialize the EDA processor with the dataset."""
        self.filepath = filepath
        self.df = self.load_data()

    def load_data(self) -> pd.DataFrame:
        """Loads the dataset from a CSV file."""
        try:
            df = pd.read_csv(self.filepath)
            print("Data successfully loaded.")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    def data_overview(self):
        """Prints an overview of the dataset, including number of rows, columns, and data types."""
        print("Dataset Overview:")
        print(self.df.info())
        print("\nSample Data:")
        print(self.df.head())

    def summary_statistics(self):
        """Prints summary statistics of the dataset."""
        print("Summary Statistics:")
        print(self.df.describe(include="all").transpose())
        return self.df.describe(include="all").transpose()
    def plot_numerical_distributions(self, numerical_features: list):
        """Plots the distribution of numerical features."""
        for feature in numerical_features:
            if feature not in self.df.columns:
                print(f"Feature {feature} not found in dataset.")
                continue

            if feature == "TransactionStartTime":
                # Convert to datetime and aggregate by date
                print(f"Processing time-series data for {feature}...")
                if not pd.api.types.is_datetime64_any_dtype(self.df[feature]):
                    self.df[feature] = pd.to_datetime(self.df[feature], errors="coerce")

                # Check for NaT values after conversion
                if self.df[feature].isna().any():
                    print(f"Warning: {feature} contains invalid datetime values.")

                self.df["TransactionDate"] = self.df[feature].dt.date
                transaction_counts = self.df.groupby("TransactionDate").size()

                # Plot time-series data
                plt.figure(figsize=(12, 6))
                transaction_counts.plot(kind="line", marker="o", color="blue")
                plt.title("Number of Transactions Over Time")
                plt.xlabel("Date")
                plt.ylabel("Transaction Count")
                plt.grid(True)
                plt.show()
            else:
                # Plot histogram for numerical features
                plt.figure(figsize=(10, 5))
                sns.histplot(self.df[feature], kde=True, bins=30, color="blue")
                plt.title(f"Distribution of {feature}")
                plt.xlabel(feature)
                plt.ylabel("Frequency")
                plt.show()

    def plot_categorical_distributions(self, categorical_features: list):
        """Plots the distribution of categorical features."""
        for feature in categorical_features:
            if feature not in self.df.columns:
                print(f"Feature {feature} not found in dataset.")
                continue

            # Check for unique value count
            unique_values = self.df[feature].nunique()
            print(f"{feature}: {unique_values} unique values.")

            # Limit categories for long categorical data
            if unique_values > 50:
                print(f"Feature {feature} has many unique values. Showing top 50 categories.")
                top_categories = self.df[feature].value_counts().head(50).index
                filtered_df = self.df[self.df[feature].isin(top_categories)]
            else:
                filtered_df = self.df

            plt.figure(figsize=(10, 5))
            sns.countplot(data=filtered_df, x=feature, order=filtered_df[feature].value_counts().index, palette="muted")
            plt.title(f"Distribution of {feature}")
            plt.xlabel(feature)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.show()
    
    def time_based_analysis(self, time_column):
        """Analyzes transaction patterns over time."""
        if time_column in self.df.columns:
            self.df[time_column] = pd.to_datetime(self.df[time_column], errors="coerce")
            self.df["Hour"] = self.df[time_column].dt.hour
            self.df["Day"] = self.df[time_column].dt.day_name()

            plt.figure(figsize=(12, 6))
            sns.countplot(data=self.df, x="Hour", palette="viridis")
            plt.title("Transactions by Hour")
            plt.xlabel("Hour of Day")
            plt.ylabel("Transaction Count")
            plt.show()

            plt.figure(figsize=(12, 6))
            sns.countplot(data=self.df, x="Day", order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], palette="muted")
            plt.title("Transactions by Day")
            plt.xlabel("Day of Week")
            plt.ylabel("Transaction Count")
            plt.show()
    def correlation_analysis(self, numerical_features: list):
        """Plots a heatmap showing correlations between numerical features."""
        valid_features = [f for f in numerical_features if f in self.df.columns]
        if not valid_features:
            print("No valid numerical features found for correlation analysis.")
            return

        correlation_matrix = self.df[valid_features].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix")
        plt.show()

    def check_missing_values(self):
        """Displays missing value counts and their percentages."""
        missing_values = self.df.isnull().sum()
        missing_percentage = (missing_values / len(self.df)) * 100
        missing_df = pd.DataFrame({"Missing Values": missing_values, "Percentage": missing_percentage})
        missing_df = missing_df[missing_df["Missing Values"] > 0]
        if missing_df.empty:
            print("No missing values found.")
        else:
            print("Missing Values:")
            print(missing_df)

    def detect_outliers(self, numerical_features: list):
        """Plots box plots for numerical features to identify outliers."""
        for feature in numerical_features:
            if feature not in self.df.columns:
                print(f"Feature {feature} not found in dataset.")
                continue

            plt.figure(figsize=(10, 5))
            sns.boxplot(data=self.df, x=feature, palette="muted")
            plt.title(f"Outliers in {feature}")
            plt.xlabel(feature)
            plt.show()
        return self.df

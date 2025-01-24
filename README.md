# Credit Risk Analysis and Modeling

This project aims to build a comprehensive understanding and modeling framework for credit risk analysis. It involves data exploration, feature engineering, credit risk classification, and deploying machine learning models for real-time predictions.

---

## Project Overview

Credit risk represents the likelihood of a borrower failing to meet financial obligations. Managing credit risk is critical for financial institutions to minimize losses while maintaining profitability. This project focuses on:

1. Understanding credit risk concepts.
2. Performing Exploratory Data Analysis (EDA) on a financial dataset.
3. Engineering relevant features for predictive modeling.
4. Constructing and evaluating machine learning models for credit scoring.
5. Developing an API to serve the trained model for real-time predictions.

---

## Project Goals

1. Analyze the dataset to identify trends, patterns, and anomalies.
2. Engineer features to enhance model performance.
3. Build a credit risk model to classify users as high or low risk.
4. Evaluate model performance using industry-standard metrics.
5. Deploy the model for real-time predictions via an API.

---

## Tasks Completed

### Task 1: Understanding Credit Risk

- Researched and documented key concepts, including:
  - Probability of Default (PD)
  - Loss Given Default (LGD)
  - Exposure at Default (EAD)
  - Alternative credit scoring methods
  - Regulatory frameworks (e.g., Basel III, IFRS 9)

### Task 2: Exploratory Data Analysis (EDA)

- **Overview of the Dataset:**

  - Explored the structure, number of rows/columns, and data types.
  - Generated summary statistics to understand the dataset's central tendency and dispersion.

- **Distribution Analysis:**

  - Visualized the distribution of numerical features to identify patterns, skewness, and outliers.
  - Analyzed categorical features to understand frequency and variability.

- **Correlation Analysis:**

  - Examined relationships between numerical variables.

- **Missing Value Analysis:**

  - Identified missing data and outlined imputation strategies.

- **Outlier Detection:**
  - Used box plots to detect and analyze outliers.

_Visual representations of these analyses are included in the project documentation._

---

## Methods and Tools Used

- **Programming Language:** Python
- **Libraries:**
  - Pandas, NumPy: Data manipulation and analysis
  - Matplotlib, Seaborn: Data visualization
  - Scikit-learn: Machine learning and feature engineering
  - Xverse, WoE: Advanced feature engineering tools
- **Key Techniques:**
  - Data normalization and standardization
  - One-hot encoding and label encoding
  - Weight of Evidence (WoE) and Information Value (IV) for feature selection

---

## Next Steps

### Task 3: Feature Engineering

- Create aggregate features (e.g., total transaction amount, average transaction amount).
- Extract temporal features (e.g., transaction hour, day, month).
- Encode categorical variables using one-hot or label encoding.
- Handle missing values through imputation or removal.
- Normalize/standardize numerical features for better model performance.

### Task 4: Default Estimator and WoE Binning

- Visualize transactions in the RFMS space to classify users as high or low risk.
- Perform Weight of Evidence (WoE) binning to prepare features for modeling.

### Task 5: Modeling

- Split the data into training and testing sets.
- Train models such as Logistic Regression, Decision Trees, Random Forest, and Gradient Boosting Machines (GBM).
- Perform hyperparameter tuning using grid search or random search.
- Evaluate models using metrics like accuracy, precision, recall, F1 score, and ROC-AUC.

### Task 6: Model Serving API Call

- Build a REST API using Flask or FastAPI to serve the trained model.
- Define endpoints to accept input data and return predictions.
- Deploy the API to a web server or cloud platform for real-time predictions.

---

## References

- [Statistical Analysis of Credit Risk](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A28n535.pdf)
- [Alternative Credit Scoring](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
- [World Bank Credit Scoring Guidelines](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf)
- [Developing a Credit Risk Model](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03)
- [Corporate Finance Institute: Credit Risk](https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/)
- [Risk Officer: Credit Risk](https://www.risk-officer.com/Credit_Risk.htm)

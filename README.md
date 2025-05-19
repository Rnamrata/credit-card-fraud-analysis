# online-payment-fraud-analysis

The project is about predicting online payment fraud. The Online Payments Fraud Detection [https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset] dataset has been collected from Kaggle. The dataset description has been provided in Kaggle.

There are seven steps of the project.
i.   Data Preprocessing
ii.  Exploratory Data Analysis (EDA) 
iii. Feature Engineering
iv.  Data Visualisation
v.   Feature Selection
vi.  Model Ealuation
vii. Final Result

## i. Data Preprocessing
Data preprocessing is the process of cleaning, transforming, and organizing raw data to make it suitable for analysis or machine learning models. The key steps for data preprocessing are: 

### 1. Initial Checks for Data Quality
The preprocessing begins with a data quality check where the dataset is examined for missing values and duplicate records. A custom function is implemented to compute the number of null values across all features and to count duplicated rows. This ensures that the dataset is clean and free from redundant or incomplete entries before further transformations are applied.
### 2. Categorical Encoding
The original transaction type column is preserved as "type_category" while being numerically encoded into "type" using Label Encoding to enable machine learning algorithms to process these categorical values.

## ii. Exploratory Data Analysis (EDA) 
Exploratory Data Analysis (EDA) is the process of examining and visualizing datasets to summarize their main characteristics. The goal of EDA is to understand the structure, patterns, and anomalies in the data, helping to make informed decisions about data cleaning, feature selection, and modeling techniques. The EDA steps applied here are:

### 1. Descriptive Statistics:
The EDA begins with a statistical overview of the dataset using the describe() function, which provides insights into the central tendencies, spread, and shape of each numerical feature. This includes metrics like mean, median, minimum, maximum, and standard deviation. 
### 2. Distribution Analysis:
Histograms of numerical features showed highly skewed distributions, particularly for transaction amounts and account balances.
### 3. Understanding Target Variable (isFraud)
To analyze class balance, the isFraud column—which indicates whether a transaction is fraudulent (1) or not (0)—is explored using value counts and a count plot. The visualization reveals a significant class imbalance between fraudulent and normal transactions, with far fewer fraudulent transactions compared to normal ones. 
### 4. Transaction Type Analysis: 
Count plots revealed the distribution of different transaction types (CASH_OUT, PAYMENT, CASH_IN, TRANSFER, DEBIT).
### 5. Correlation Analysis: 
A heatmap between fraud and transaction types demonstrated that certain transaction types (like TRANSFER and CASH_OUT) were more associated with fraudulent activities. This targeted insight helps narrow the focus for fraud detection models and investigations.
### 6. Categorical Feature Comparison: 
Box and violin plots compared key features (amount, balances) between fraudulent and legitimate transactions, revealing significant differences in their distributions.

The analysis shows that fraudulent transactions typically involve high amounts and originate from accounts with large balances, which are often drained completely. The destination accounts usually have low initial balances, suggesting that fraudsters use new or inactive accounts to avoid detection and maximize their gains. 

## iii. Feature Engineering
Feature engineering is the process of creating, transforming, or selecting data features to improve the performance of machine learning models. 

The analysis reveals that fraudulent transactions are limited to only two types: cashout (4100 cases) and transfer (4097 cases), while the remaining types—payment, debit, and cash_in—show no instances of fraud. This suggests that fraudsters tend to target cashout and transfer methods, likely because these transaction types offer easier ways to move or withdraw funds.

### 1. Transaction Type Weighting: 
Assigning different weights to transaction types based on their fraud likelihood.
### 2. Amount Binning: 
Creating categorical bins for transaction amounts and calculating fraud risk for each bin.
### 3. Scaling Techniques: 
Implementing multiple scaling methods (MinMax, Standard, Robust) to normalize features, with clear visualization of their effects. After analyzing their comparative effects to evaluate the impact of each method on data distribution and outlier handling Robust method outperfomed the others
### 4. Log Transformations: 
To handle skewed distributions and reduce the impact of large outliers, log transformations are applied to some of the balance difference features. This helps normalize the data and makes patterns more discernible for algorithms sensitive to scale.
### 5. Balance Differentials: 
Creating new features based on the difference between old and new balances for both origin and destination accounts (oldbalanceOrg, newbalanceOrig, oldbalanceDest, and newbalanceDest). These differences help capture whether money actually moved during the transaction, which is a strong indicator in fraud detection.
### 6. Scaling New Features:
Certain features, specifically balance difference metrics, are log-transformed to reduce skewness and stabilize variance. Both the original and log-transformed versions (Org_balance_diff, Dest_balance_diff, Org_balance_diff_log, Dest_balance_diff_log) are selected for scaling. Prior to applying the transformations, a check ensures that these columns exist in the dataset to avoid runtime issues. This step helps in improving model performance by normalizing data and reducing the influence of extreme values.

The feature engineering process resulted in a significantly enhanced dataset with transformed features that better capture fraud patterns.

## iv. Data Visualisation
The visualisation part confirms that fraud detection effectiveness can be significantly enhanced by examining the complete transaction flow rather than isolated metrics. After analyzing transaction patterns across multiple dimensions, several distinctive fraud indicators emerge:

### 1. Account Selection Patterns: 
Fraudsters strategically target high-balance origin accounts and funnel funds into previously dormant or zero-balance destination accounts.
### 2. Complete Account Draining: 
Fraudulent transactions typically empty origin accounts entirely, leaving near-zero balances, unlike legitimate transactions that maintain varied remaining balances.
### 3. Mathematical Inconsistencies: 
The relationship between transaction amounts and resulting balance changes shows irregularities in fraudulent cases, with inconsistent patterns in both origin and destination accounts.
### 4. Transformation Benefits: 
Log transformations reveal otherwise hidden patterns by normalizing skewed distributions and reducing the impact of extreme values, making anomaly detection more effective.

The most reliable fraud indicators combine account balance characteristics with mathematical relationship anomalies between transaction amounts and balance changes. Specifically, transactions that (1) drain high-balance accounts completely, (2) deposit into previously inactive accounts, and (3) show inconsistent relationships between transaction amounts and balance changes warrant heightened scrutiny. These patterns, particularly when analyzed using appropriate transformations to manage data skewness, provide financial institutions with powerful tools to identify potentially fraudulent transactions before they can be completed.

## v. Feature Selection
In the feature selection part of the analysis, irrelevant and redundant features such as nameOrig, nameDest, and isFlaggedFraud are removed, as they do not contribute meaningfully to fraud detection. Additionally, new features like balance differences (Org_balance_diff, Dest_balance_diff) and their log-transformed versions are created to capture transaction behavior more effectively. This refined feature set helps improve model performance by focusing on variables that show clear patterns and distinctions between fraudulent and non-fraudulent transactions.

## vi. Model Evaluation



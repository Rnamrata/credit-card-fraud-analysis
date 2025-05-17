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
Data preprocessing is the process of cleaning, transforming, and organizing raw data to make it suitable for analysis or machine learning models. There are three key steps for data preprocessing.

### 1. Initial Checks for Data Quality

The preprocessing begins with a data quality check where the dataset is examined for missing values and duplicate records. A custom function is implemented to compute the number of null values across all features and to count duplicated rows. This ensures that the dataset is clean and free from redundant or incomplete entries before further transformations are applied.

### 2. Feature Scaling

To normalize the numerical features, especially those with varying units and ranges, multiple scaling techniques are employed. The transaction_type_weight column is scaled using MinMaxScaler to bring its values between 0 and 1. Additionally, the amount feature undergoes scaling using three different methods—MinMaxScaler, StandardScaler, and RobustScaler—to analyze their comparative effects. The results are visualized using a box plot to evaluate the impact of each method on data distribution and outlier handling.

### 3. Log Transformation and Scaling

Certain features, specifically balance difference metrics, are log-transformed to reduce skewness and stabilize variance. Both the original and log-transformed versions (Org_balance_diff, Dest_balance_diff, Org_balance_diff_log, Dest_balance_diff_log) are selected for scaling. Prior to applying the transformations, a check ensures that these columns exist in the dataset to avoid runtime issues. This step helps in improving model performance by normalizing data and reducing the influence of extreme values.


## ii. Exploratory Data Analysis (EDA) 
Exploratory Data Analysis (EDA) is the process of examining and visualizing datasets to summarize their main characteristics. The goal of EDA is to understand the structure, patterns, and anomalies in the data, helping to make informed decisions about data cleaning, feature selection, and modeling techniques. The EDA steps applied here are:

### 1. Descriptive Statistics and Distribution Analysis

The EDA begins with a statistical overview of the dataset using the describe() function, which provides insights into the central tendencies, spread, and shape of each numerical feature. This includes metrics like mean, median, minimum, maximum, and standard deviation. To complement this, a comprehensive set of histograms is plotted to visualize the distribution of these features. These histograms help identify skewness, outliers, and the need for transformations such as normalization or log-scaling in subsequent steps.

### 2. Understanding Target Variable (isFraud)

To analyze class balance, the isFraud column—which indicates whether a transaction is fraudulent (1) or not (0)—is explored using value counts and a count plot. The visualization reveals a significant class imbalance, with far fewer fraudulent transactions compared to normal ones. This is a critical insight for model development, as it suggests that strategies like resampling, class weighting, or anomaly detection may be required to handle the imbalance effectively.

### 3. Analyzing Categorical Features (type and type_catagory)

The distribution of transaction types is examined using histograms and count plots. First, the type column, which categorizes the nature of each transaction (e.g., PAYMENT, TRANSFER), is visualized to understand its spread and frequency. Then, a separate categorical mapping column, type_catagory, is analyzed in a similar manner. These visualizations provide an understanding of user behavior and transaction trends, while also helping to identify which transaction types may be more frequently associated with fraud. A heatmap visualizes the relationship between transaction type and fraud occurrence. It shows that fraudulent activity is confined to TRANSFER and CASH_OUT transactions, with no fraud detected in PAYMENT, DEBIT, or CASH_IN types. This targeted insight helps narrow the focus for fraud detection models and investigations.

The analysis shows that fraudulent transactions typically involve high amounts and originate from accounts with large balances, which are often drained completely. The destination accounts usually have low initial balances, suggesting that fraudsters use new or inactive accounts to avoid detection and maximize their gains. 

## iii. Feature Engineering
Feature engineering is the process of creating, transforming, or selecting data features to improve the performance of machine learning models. The feature engineering is focused on creating informative, numerical features from raw transactional data to improve the accuracy and efficiency of machine learning models for fraud detection.

The analysis reveals that fraudulent transactions are limited to only two types: cashout (4100 cases) and transfer (4097 cases), while the remaining types—payment, debit, and cash_in—show no instances of fraud. This suggests that fraudsters tend to target cashout and transfer methods, likely because these transaction types offer easier ways to move or withdraw funds.

### 1. Categorical Encoding

A new feature, type_catagory, is created as a copy of the original type column to preserve its categorical labels. The original type column is then encoded using Label Encoding, converting the textual transaction types (like “TRANSFER”, “CASH_OUT”) into numeric values. This step is essential because machine learning algorithms generally require numerical input for processing.

### 2. Derived Features

The dataset includes transactional balance columns such as oldbalanceOrg, newbalanceOrig, oldbalanceDest, and newbalanceDest. These are used to create meaningful derived features like the difference between original and new balances for both the origin and destination accounts. These differences help capture whether money actually moved during the transaction, which is a strong indicator in fraud detection.

### 3. Log Transformations

To handle skewed distributions and reduce the impact of large outliers, log transformations are applied to some of the balance difference features. This helps normalize the data and makes patterns more discernible for algorithms sensitive to scale.

## iv. Data Visualisation

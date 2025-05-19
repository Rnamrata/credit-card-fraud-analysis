# online-payment-fraud-analysis

The project is about predicting online payment fraud. The Online Payments Fraud Detection [https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset] dataset has been collected from Kaggle. The dataset description has been provided in Kaggle.

There are eight steps of the project.
i.    Data Preprocessing
ii.   Exploratory Data Analysis (EDA) 
iii.  Feature Engineering
iv.   Data Visualisation
v.    Feature Selection
vi.   Model Ealuation
vii.  Best Model Selection and Comparision
viii. Final Result

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

A correlation heatmap visualizes the relationships between these selected features and the target variable (isFraud), confirming that the engineered features show stronger correlations with fraudulent activities than many original features. The selected feature set is saved as 'clean_credit_card_fraud_dataset.csv' for model training.

## vi. Model Evaluation
Various methods were introduced for the evaluation perpose. The 'evaluate_model' used to  predict the fraudulent and establishing initial performance benchmarks across different metrics such as accuracy, precision, recall, F1-score, ROC AUC, MSE, and RMSE. The 'get_model_scores' measured the train and validation (test) score. The 'assess_fit' method assessed the model performence. Several charts, maps, and graphs have been ploted to compare and analyse the results of each models and find the best one. The analysis employed a comprehensive approach to model evaluation with three phases. 

The dataset is divided into training and testing subsets through a generalised spilt that allowed the model to learn from the training data and then be evaluated on unseen test data, providing a more realistic assessment of how the model would perform in real-world scenarios. 

Afterwards, to address data challenges, a comprehensive balancing and cleaning strategy has been implemented by duplicating minority fraud cases through random oversampling (RandomOverSampler) and potentially generating synthetic examples using synthetic oversampling techniques (SMOTE), while handling missing values through imputation. An advanced data splitting technique was used to improved model's ability to detect the rare fraudulent transactions without compromising overall performance for Grid search and Bayes search pipeline.

### 1. General Pipeline:
The general pipeline represents a standard model training approach and classification models without hyperparameter tuning. It establishes a performance baseline using default parameters. Initially, models trained on the imbalanced dataset (without resampling) struggle to detect fraud accurately due to the overwhelming number of non-fraudulent transactions. Several models i.e. Logistic Regression, Random Forest, Support Vector Machine (SVM), LightGBM, XGBoost were evaluated using a standard pipeline.

### 2. Grid Search Cross Validation Pipeline:
The Grid Search pipeline involves a systematic search over a predefined set of hyperparameters using cross-validation to optimize model performance. It performs best when computational resources allow for exhaustive testing and when reliable generalization is critical. Models like Random Forest, LightGBM, Gradient Boosting, HistGradient Boosting, XGBoost, SGDClassifier were tuned with this pipeline. This phase fine-tuned each model to find its optimal hyperparameters.

### 3. Bayes Search Cross Validation Pipeline:
The Bayes Search pipeline uses Bayesian optimization for more efficient hyperparameter tuning compared to Grid Search. It evaluates fewer combinations while still identifying high-performing settings. This approach is computationally faster and often achieves better or comparable results. This pipeline provides robust models with improved generalization, especially in detecting rare fraudulent cases. It strikes a balance between performance and efficiency, making it well-suited for imbalanced classification tasks. The same models like the Grid search were tuned with this pipeline with ontinuous parameter ranges and optimal hyperparameters more efficiently.

Throughout all phases, the code evaluates model fitting characteristics by measuring the gap between training and validation scores to identify overfitting or underfitting. Visualizations include bar charts comparing metric performance, radar charts showing normalized performance across dimensions, heatmaps of key metrics, ROC AUC comparisons, and confusion matrices. This rigorous evaluation framework ensures that model selection is based on comprehensive performance assessment rather than a single metric.

## vii.  Best Model Selection and Comparision
The fraud detection system employed a rigorous selection methodology comparing top performers from three optimization approaches: baseline models, grid search optimization, and Bayesian optimization. Selection prioritized well-fitted models with minimal overfitting, using a custom function that balanced validation scores (higher being better) with overfit gaps (smaller being better). This approach was complemented by comprehensive visualizations including comparative metric bar charts, radar plots of normalized performance, side-by-side confusion matrices, and trend analyses across evaluation metrics, providing a multidimensional view of model performance.

### 1. Performance Metrics Comparison: 
A comprehensive comparison of three fraud detection models (XGBoost, LightGBM, and Gradient Boosting) across five critical performance metrics (accuracy, precision, recall, F1 score, ROC AUC), with each model having versions from different optimization sources (baseline, grid search, and Bayesian optimization).

The visualization reveals that both LightGBM and Gradient Boosting consistently achieve near-perfect scored (close to 1.0) on accuracy, precision, recall, F1 score, and ROC AUC when optimized through grid search or Bayesian methods. XGBoost showed lower performance, particularly in the baseline implementation (results_df). The most dramatic improvements were seen in the precision and ROC AUC metrics, where optimization techniques elevated performance from approximately 0.8 to nearly 1.0.

### 2. Normalized Performance Radar Chart:
The radar chart visually confirms Gradient Boosting and LightGBM as superior fraud detection models, displaying nearly identical perfect pentagons across all five metrics. Their balanced excellence stands in stark contrast to XGBoost's significantly weaker performance, which is barely visible on the chart. This visualization effectively illustrates why Gradient Boosting narrowly edges out LightGBM as the optimal model, with both demonstrating the balanced precision-recall performance essential for effective fraud detection.

### 3. Confusion Matrix: 
The confusion matrices provided deeper insight into each model's classification decisions:

XGBoost demonstrated significant false positives (350,878 legitimate transactions incorrectly flagged as fraudulent) and false negatives (248 fraudulent transactions missed), explaining its lower overall performance.

LightGBM showed excellent performance with only 51 false positives and 24 false negatives, achieving a near-perfect balance between fraud detection and minimizing false alerts.

Gradient Boosting performed similarly to LightGBM with 36 false positives and 24 false negatives, indicating exceptional precision in fraud identification while maintaining high recall.

### 4. Accuracy Across Optimization Methods:
Model accuracy across different optimization approaches, clearly demonstrating the impact of hyperparameter tuning. XGBoost's baseline accuracy (~0.72) was substantially lower than its optimized versions. LightGBM with grid search optimization and Gradient Boosting reached perfect accuracy (~1.0) with Bayesian optimization. Both grid search and Bayesian optimization deliver comparable performance improvements.

### 5. Classification Reports:
The classification reports revealed significant performance disparities between models. XGBoost showed perfect precision for legitimate transactions but completely failed with fraudulent ones (0.00 precision), resulting in a poor fraud F1-score of 0.01 despite reasonable recall (0.85). In stark contrast, both LightGBM and Gradient Boosting achieved perfect 1.00 scores across all metrics (precision, recall, F1-score) for both transaction classes, demonstrating flawless classification balance. Their identical performance extended to perfect accuracy, macro averages, and weighted averages of 1.00 across approximately 24,591 transactions, including 8,197 fraudulent cases. This ideal performance indicated these models can reliably detect fraud without generating false alerts—a critical requirement for practical fraud detection systems.

## viii. Final Result
The final selection concluded by computing the average performance across key metrics (accuracy, precision, recall, F1 score, ROC AUC) for each model. The model with the highest average score was selected as the overall best model. This thorough comparison ensured that the selected model offered the optimal balance of fraud detection accuracy while minimizing both false positives and false negatives, which arweree critical considerations for a practical fraud detection system. 

The final results showed the average performance scores for each model:

    1. Gradient Boosting: 0.997309 (highest score)
    2. LightGBM: 0.996631 (very close second)
    3. XGBoost: 0.479314 (significantly lower)

Based on this comprehensive evaluation across multiple metrics, Gradient Boosting with Bayesian optimization was identified as the best-fitting model with an impressive average score of 0.997, narrowly outperforming LightGBM by about 0.0007. Both models demonstrated near-perfect performance, while XGBoost significantly underperformed with an average score of only 0.479.

This selection represented the culmination of the entire analysis process, indicating that the Gradient Boosting algorithm, after proper optimization, provided the most balanced and effective approach for detecting fraudulent transactions while minimizing false positives and false negatives.
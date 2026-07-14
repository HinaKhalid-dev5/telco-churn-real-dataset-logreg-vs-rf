
CUSTOMER CHURN PREDICTION DASHBOARD

PROJECT OVERVIEW

This project analyzes customer behavior data from a telecom 
company to predict customer churn (whether a customer will 
leave the company or not). The project covers the complete 
machine learning pipeline from raw data to business insights.

Dataset: Telco Customer Churn dataset (7032 customers, 
21 original features)


WHAT THIS PROJECT INCLUDES

1. Data Analysis
   - Data cleaning (handled missing values, fixed data types)
   - Exploratory Data Analysis with 10+ visualizations
   - Summary statistics

2. Feature Engineering
   - Created 5 new behavioral features:
     TotalServices, TenureGroup, AvgMonthlySpend, 
     SupportCount, SpendTrend

3. Machine Learning Models
   - Logistic Regression and Random Forest trained
   - Evaluated using Accuracy, Precision, Recall, 
     F1 Score, and ROC-AUC
   - Logistic Regression selected as final model 
     (ROC-AUC: 0.833)

4. Customer Segmentation
   - Customers segmented into High, Medium, Low value tiers
   - Based on tenure, monthly charges, and service usage

5. Churn Prediction System
   - Probability-based churn prediction for every customer
   - Risk categorization (Low/Medium/High Risk)

6. Business Insights Report
   - Top churn drivers identified
   - Revenue impact analysis ($531,817 annual revenue at risk)
   - Actionable retention recommendations

7. Bonus Features
   - SHAP explainability for transparent model interpretation
   - Interactive Streamlit dashboard for non-technical users



================================================================
HOW TO RUN THIS PROJECT
================================================================

REQUIREMENTS:
- Python 3.x installed
- Required libraries: pandas, numpy, matplotlib, seaborn,
  scikit-learn, shap, streamlit

INSTALL REQUIRED LIBRARIES:
Open terminal/command prompt and run:

    pip install pandas numpy matplotlib seaborn scikit-learn 
    shap streamlit

TO RUN THE JUPYTER NOTEBOOK:
1. Open the "notebook" folder
2. Open churn_prediction.ipynb in VS Code or Jupyter
3. Run all cells in order from top to bottom

TO RUN THE INTERACTIVE DASHBOARD:
1. Open terminal in VS Code
2. Navigate to the "dashboard" folder:
   cd dashboard
3. Run this command:
   streamlit run app.py
4. Dashboard will open automatically in your web browser
5. Enter customer details and click "PREDICT CHURN" to see 
   results

================================================================
KEY RESULTS SUMMARY
================================================================

- Dataset size: 7032 customers (after cleaning)
- Overall churn rate: 26.5%
- Best model: Logistic Regression
- Model Accuracy: 78.7%
- Model ROC-AUC: 0.833
- High risk customers identified: 537
- Annual revenue at risk: $531,817

Top 3 churn drivers (confirmed by SHAP analysis):
1. Tenure (newer customers churn most)
2. Contract type (month-to-month customers churn most)  
3. Monthly/Total charges (higher charges increase churn risk)


KEY LEARNINGS

Through this project I learned the complete machine learning 
workflow including data cleaning, exploratory analysis, 
feature engineering, model training and evaluation, and 
translating technical results into business recommendations. 
I learned why accuracy alone is misleading for imbalanced 
datasets and the importance of precision, recall, and F1 
score. I also learned how SHAP values provide more reliable 
feature importance than raw model coefficients, and built 
my first interactive web dashboard using Streamlit.


CHALLENGES FACED


The biggest challenge was understanding evaluation metrics 
(precision, recall, F1, ROC-AUC) conceptually rather than 
just calculating them. I also faced technical challenges 
running the Streamlit dashboard correctly through the 
terminal rather than the standard Python run button.

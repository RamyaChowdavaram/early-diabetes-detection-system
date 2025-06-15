# Early Diabetes Detection System (EDDS)

## Executive Summary
Diabetes is a chronic and potentially life-threatening disease that affects millions of people worldwide. Early diagnosis plays a critical role in preventing complications, improving quality of life, and reducing healthcare costs. However, traditional diagnostic methods often rely on delayed clinical assessments or invasive procedures, which can result in late detection and missed treatment opportunities.

The **Early Diabetes Detection System (EDDS)** aims to address this issue by developing a **predictive machine learning model** capable of identifying individuals at risk of diabetes based on easily accessible patient health data (such as glucose levels, BMI, blood pressure, age, etc).

### Goals

- Build a **robust and accurate classification model** that can predict the predict the and assist in the early detection of diabetes using patient health data. 
- Prioritize **early detection**, focusing on models that demonstrate strong **recall (sensitivity)** and **F1 score**, ensuring diabetic cases are not missed.
- Develop a model that can be **integrated into clinical workflows** as a decision-support tool for healthcare professionals.

### Research Question
Can a machine learning classification model accurately predict the likelihood of an individual having diabetes based on medical and lifestyle, demographic related input features?

### Challenges

- Ensuring the model maintains **high recall** without triggering too many **false positives** (over-diagnosing non-diabetic patients).
- Dealing with **imbalanced or noisy health data**, which is common in medical datasets.
- Making the model **interpretable** enough for clinicians to trust and act upon its predictions.
- Guaranteeing **fairness** and **generalizability** across different patient demographics (ex: age groups, genders, ethnicity)

### Rationale

Diabetes is a global health crisis affecting millions of people. Often without symptoms until serious complications like heart disease, kidney failure, or blindness occur. Many cases go undiagnosed until it's too late for preventive treatment.
If this issue is not addressed, individuals will continue to suffer avoidable health problems, and healthcare systems will be burdened with high treatment costs for advanced disease.
Early prediction allows people to take action before complications develop. By using machine learning to flag high-risk individuals based on simple health indicators, we can support:
  - Doctors with faster, data-driven decision-making
  - Common people with awareness of their personal risk and when to seek help
  - Healthcare systems with automated screening to prioritize early intervention
Ultimately, this analysis brings significant public health value. It empowers prevention, reduces healthcare costs, and improves long-term outcomes for individuals and society.

## Model Outcomes or Predictions

- **Learning Type**: **Supervised Learning**
- **Model Category**: **Classification**
- **Reason**: Each data instance (patient) in the training dataset is labeled as either **diabetic** or **non-diabetic**, which allows the model to learn from known outcomes.

- The model predicts a **binary outcome**:
  - 1 → The patient is **diabetic**
  - 0 → The patient is **non-diabetic**

### Models Used

The following supervised classification algorithms were evaluated:

| Model                | Learning Type | Output Type      |
|----------------------|---------------|------------------|
| Random Forest         | Supervised    | Binary Class (0/1) |
| Decision Tree         | Supervised    | Binary Class (0/1) |
| Support Vector Machine (SVM) | Supervised | Binary Class (0/1) |
| Logistic Regression   | Supervised    | Binary Class (0/1) |
| K-Nearest Neighbors (KNN) | Supervised | Binary Class (0/1) |
| Naive Bayes           | Supervised    | Binary Class (0/1) |

- The system uses performance metrics like **accuracy, recall, F1 score, and ROC-AUC** to assess prediction quality.
  
## Data Sources
https://www.kaggle.com/datasets/asinow/diabetes-dataset

The data should include various health parameters, lifestyle habits, and genetic predispositions that contribute to diabetes risk. The data should be structured with realistic distributions. The dataset should be useful for exploring the relationships between lifestyle choices, genetic factors, and diabetes risk, providing valuable insights for predictive modeling and health analytics. The data should include some or most of the below features:

- Age: The age of the individual (18-90 years).
- Pregnancies: Number of times the patient has been pregnant.
- BMI (Body Mass Index): A measure of body fat based on height and weight (kg/m²).
- Glucose: Blood glucose concentration (mg/dL), a key diabetes indicator.
- BloodPressure: Systolic blood pressure (mmHg), higher levels may indicate hypertension.
- HbA1c: Hemoglobin A1c level (%), representing average blood sugar over months.
- LDL (Low-Density Lipoprotein): "Bad" cholesterol level (mg/dL).
- HDL (High-Density Lipoprotein): "Good" cholesterol level (mg/dL).
- Triglycerides: Fat levels in the blood (mg/dL), high values increase diabetes risk.
- WaistCircumference: Waist measurement (cm), an indicator of central obesity.
- WHR (Waist-to-Hip Ratio): Waist circumference divided by hip circumference.
- FamilyHistory: Indicates if the individual has a family history of diabetes (1 = Yes, 0 = No).
- DietType: Dietary habits (0 = Unbalanced, 1 = Balanced, 2 = Vegan/Vegetarian).
- Hypertension: Presence of high blood pressure (1 = Yes, 0 = No).
- MedicationUse: Indicates if the individual is taking medication (1 = Yes, 0 = No).
- Outcome: Diabetes diagnosis result (1 = Diabetes, 0 = No Diabetes).

![Diabetes Visualization](https://raw.githubusercontent.com/RamyaChowdavaram/early-diabetes-detection-system/main/data/images/diabetes_1.png)
![Diabetes Visualization](https://raw.githubusercontent.com/RamyaChowdavaram/early-diabetes-detection-system/main/data/images/diabetes_2.png)
![Diabetes Visualization](https://raw.githubusercontent.com/RamyaChowdavaram/early-diabetes-detection-system/main/data/images/diabetes_3.png)


### Data Preprocessing / Preparation

#### Dataset Overview
- **Technique Used**:  df.shape ,  df.info() 
- **Result**:
  - Records: 9,538 rows
  - Features: 17 variables (continuous and categorical)
  - Memory Usage: ~1.2 MB
  - Target Variable:  Outcome  (Binary: 0 = No Diabetes, 1 = Diabetes)
  - Conclusion: Dataset size is suitable for machine learning with good generalization potential.

#### Missing Value Detection
- **Technique Used**:  df.isnull().sum() 
- **Result**: No missing values were detected in any of the features.
  - Continuous variables (e.g.,  bmi ,  glucose ,  hdl ) are fully populated.
  - Categorical variables (e.g.,  familyhistory ,  diettype ) are complete.
  - Conclusion: No imputation was required, which simplifies preprocessing and ensures reliable model input.

#### Duplicate Row Detection
- **Technique Used**:  df.duplicated().sum() 
- **Result**: No duplicate rows found in the dataset.
  - Conclusion: Ensures data integrity and avoids data leakage during training.

#### Data Type Verification
- **Technique Used**:  df.dtypes 
- **Result**:
  -  float64  for continuous variables like  bmi ,  glucose ,  ldl , etc.
  -  int64  for categorical or discrete variables like  age ,  familyhistory ,  outcome .
  - Conclusion: Correct data types ensure compatibility with machine learning algorithms and statistical functions.

#### Column Name Standardization
- **Technique Used**:  df.columns = df.columns.str.lower() 
- **Result**: All column names were converted to lowercase.
  - Prevents case-sensitivity issues in downstream processing or code.

#### Categorical Label Cleaning
- **Technique Used**:  .replace()  or  .apply()  on text-based columns
- **Result**: Inconsistent categorical values such as  'N/A'  and  'Not Applicable'  were standardized to  'not applicable' .
  - Conclusion: This improves consistency and prevents issues during encoding.

#### Feature Distribution and Variability Assessment
- **Technique Used**:  df.describe() ,  .nunique() , visualizations (e.g., histograms or boxplots)
- **Results**:
  - **Continuous Features**: High variability observed (e.g.,  bmi  with 2,378 unique values,  glucose ,  ldl , and  triglycerides ).
  - **Categorical/Binary Features**:  familyhistory ,  hypertension , and  outcome  have 2 unique values — ideal for classification tasks.
  - **Moderate Granularity**:  age  spans 18–89 years (72 unique values),  pregnancies  ranges from 0 to 16.
  - Conclusion: The dataset includes both high-variability continuous features and clean categorical variables, providing a strong basis for model training.

#### Outlier and Anomaly Detection
- **Technique Used**:  .describe() , visual methods (e.g., boxplots), logical checks for invalid ranges
- **Result**: Some negative or implausible values found in  ldl  and  hdl .
  - These should be reviewed, corrected, or removed during further preprocessing to ensure model robustness.

#### Feature Distribution & Variability
- **Continuous Features**:
  - High variability observed in features like  bmi  (2,378 unique values),  glucose ,  ldl , and  triglycerides .
  - These variables provide rich granularity for prediction.
- **Categorical/Binary Features**:
  - Features such as familyhistory, hypertension, and outcome have 2 unique values each, ideal for classification.
  - diettype has low cardinality and is suitable for encoding.
- **Other Notables**:
  - age shows a good spread (18–89 years; 72 unique values).
  - pregnancies ranges from 0 to 16, reflecting a diverse population.

#### Outliers and Anomalies
- While most clinical features fall within expected ranges, some **outliers and negative values** were observed in  ldl  and  hdl . These should be checked and addressed to improve model robustness.

#### Train-Test Split
- The dataset was split using an **80/20 stratified approach** to preserve class balance:
   python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
   
#### Encoding
- No categorical encoding was needed in the final selected features:
  - All selected variables were already numeric ( int64  or  float64 )
  - Binary variables (e.g.,  familyhistory ,  medicationuse ) were already encoded as 0/1

#### Explore & Understand the Data (Exploratory Data Analysis) - Pandas, numpy, matlib, plotly, seaborn
  - Understand the types of variables (numerical, categorical).
  - Explore the shape of the data (rows and columns).
  - Check for missing values, duplicates, or incorrect data types.
  - **Visualize** the data to identify trends, outliers, and correlations between features.
  - Check basic statistics like mean, median, standard deviation, and distribution of data.
  - Use histograms, box plots, and pair plots to visualize data relationships.
  
## Modelling
   - Logistic Regression
   - Random Forest
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
   - Decision Tree
   - Naive Bayes

- Evaluation Metrics: Accuracy, F1 Score, ROC-AUC, Confusion Matrix
- Model Selection: GridSearchCV with StratifiedKFold
- Each model was tuned using **GridSearchCV** for optimal hyperparameters and evaluated using cross-validation and final test performance.

## Model Evaluation

#### Evaluation Results (on test data)

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| **Random Forest**   | 1.00     | 1.000     | 1.000  | 1.00     | 1.00    |
| **Decision Tree**   | 1.00     | 1.000     | 1.000  | 1.00     | 1.00    |
| **SVM (Linear)**    | 1.00     | 0.998     | 0.997  | 1.00     | 1.00    |
| **Logistic Regression** | 1.00 | 1.000     | 0.994  | 1.00     | 1.00    |
| **KNN**             | 0.98     | 0.990     | 0.954  | 0.97     | 1.00    |
| **Naive Bayes**     | 0.96     | 1.000     | 0.882  | 0.94     | 1.00    |

#### Cross-Validation Results (mean scores across folds)

| Model               | Accuracy | F1 Score | ROC-AUC |
|---------------------|----------|----------|---------|
| **Decision Tree**   | 1.000    | 1.000    | 1.000   |
| **SVM**             | 0.999    | 0.998    | 0.9999  |
| **Logistic Regression** | 0.998 | 0.998    | 0.9999  |
| **Random Forest**   | 0.998    | 0.997    | 1.000   |
| **KNN**             | 0.983    | 0.975    | 0.995   |
| **Naive Bayes**     | 0.958    | 0.935    | 0.998   |


Evaluated models based on five key metrics:

- **Accuracy**: Overall how often the model was correct.
- **Precision**: Of all patients flagged as diabetic, how many were actually diabetic.
- **Recall**: Of all actual diabetic cases, how many did the model detect (very important).
- **F1 Score**: A balance between precision and recall.
- **ROC-AUC**: A measure of how well the model distinguishes between diabetic and non-diabetic cases.
  
![Diabetes Visualization](https://raw.githubusercontent.com/RamyaChowdavaram/early-diabetes-detection-system/main/data/images/diabetes_4.png)
![Diabetes Visualization](https://raw.githubusercontent.com/RamyaChowdavaram/early-diabetes-detection-system/main/data/images/diabetes_5.png)

- **Random Forest** and **Decision Tree** achieved **perfect accuracy, recall, and F1 scores**, suggesting they performed flawlessly on the test set.
- **SVM** and **Logistic Regression** also performed **extremely well**, with minor drops in recall but still excellent overall reliability.
- **KNN** showed high accuracy but slightly lower recall, meaning it missed some diabetic cases.
- **Naive Bayes**, while fast and efficient, had the **lowest recall**, meaning it failed to detect a higher number of true diabetic cases — not ideal in healthcare settings.

#### What Do These Results Mean?

In a **healthcare context**, the most important metric is **recall** - we don’t want to miss identifying patients who actually have diabetes.

- Models with **low recall** may lead to **undiagnosed patients**, which can result in serious health consequences.
- Models with **high precision** help reduce false alarms, ensuring non-diabetic patients aren't wrongly flagged.

We prefer models that **maximize recall without sacrificing precision** — hence models like **Random Forest, Decision Tree** are ideal.

#### Recommendations

- **Use Random Forest** or **Decision Tree** if performance is the top priority.
  - Perfect scores on all metrics.
  - Decision Tree is also **easy to interpret**, which helps explain predictions to healthcare professionals.
- **SVM** and **Logistic Regression** are also great options when simplicity or linear relationships are preferred.
- **Avoid Naive Bayes** for this task, due to its tendency to miss actual diabetic patients.
- **KNN** is acceptable but may require tuning or feature scaling for optimal results.

## Next Steps

##### Model Deployment

- **API Integration**: Convert the trained model into an API using tools like Flask, FastAPI, or Django REST Framework.
- **Cloud Hosting**: Deploy the model via platforms like AWS, Azure, GCP, or Heroku for real-time accessibility.
- **Frontend Integration**: Build a simple UI (web or mobile) for healthcare professionals to input patient data and view predictions.

##### Model Monitoring & Maintenance

- **Performance Tracking**: Continuously monitor metrics such as accuracy, precision, and recall on real-world data.
- **Drift Detection**: Implement checks to detect changes in input data distribution (data drift) or model output performance (concept drift).
- **Re-training Pipelines**: Automate re-training with updated data to maintain relevance and accuracy over time.

##### Data Privacy & Compliance

- **Data Encryption**: Secure user data at rest and in transit using encryption standards.
- **Anonymization**: Remove or mask personally identifiable information (PII) where applicable.

##### Clinical Validation

- **Pilot Testing**: Collaborate with healthcare professionals to test the system in clinical settings.
- **Medical Review**: Ensure that model decisions are reviewed and validated by certified practitioners.
- **Regulatory Approval**: For clinical use, approval from medical regulatory bodies may be required (e.g., FDA, CE).

##### calability & Optimization

- **Batch Processing**: Enable the system to handle large volumes of patient data efficiently.
- **Latency Optimization**: Reduce response time for real-time predictions through model compression or caching.

##### User Training & Documentation

- **Clinician Training**: Educate end-users (e.g., doctors, nurses) on how to use and interpret the tool.
- **User Manual**: Provide clear documentation, tooltips, and example cases.
                                                 
## Outline of project
- Jupyter Notebook : https://github.com/RamyaChowdavaram/early-diabetes-detection-system/blob/main/diabetes_prediction_model.ipynb
- Diabetes dataset : https://github.com/RamyaChowdavaram/early-diabetes-detection-system/tree/main/data

## Contact and Further Information
Ramya Chowdavaram

Email: ramya.chvm@gmail.com

https://www.linkedin.com/in/ramya-c-9aa161234/


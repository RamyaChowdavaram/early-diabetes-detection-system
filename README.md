# Early Diabetes Detection System (EDDS) 

**Ramya Chowdavaram**

#### Executive summary
This project aims to develop a predictive model that can assist in the early detection of diabetes using patient health data. 
By leveraging machine learning classification techniques, we achieve high accuracy in identifying diabetic individuals. With strong performance in recall and F1 score, the system shows potential as a valuable decision-support tool in healthcare for early diagnosis and risk screening.

#### Rationale
Diabetes is a global health crisis affecting millions of people. Often without symptoms until serious complications like heart disease, kidney failure, or blindness occur. Many cases go undiagnosed until it's too late for preventive treatment.
If this issue is not addressed, individuals will continue to suffer avoidable health problems, and healthcare systems will be burdened with high treatment costs for advanced disease.
Early prediction allows people to take action before complications develop. By using machine learning to flag high-risk individuals based on simple health indicators, we can support:
  - Doctors with faster, data-driven decision-making
  - Common people with awareness of their personal risk and when to seek help
  - Healthcare systems with automated screening to prioritize early intervention
Ultimately, this analysis brings significant public health value. It empowers prevention, reduces healthcare costs, and improves long-term outcomes for individuals and society.

#### Research Question
Can a machine learning classification model accurately predict the likelihood of an individual having diabetes based on medical and lifestyle, demographic related input features?
    
#### Data Sources
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
                                                                                   
#### Methodology
##### Explore & Understand the Data (Exploratory Data Analysis) - Pandas, numpy, matlib, plotly, seaborn
  - Understand the types of variables (numerical, categorical).
  - Explore the shape of the data (rows and columns).
  - Check for missing values, duplicates, or incorrect data types.
  - **Visualize** the data to identify trends, outliers, and correlations between features.
  - Check basic statistics like mean, median, standard deviation, and distribution of data.
  - Use histograms, box plots, and pair plots to visualize data relationships.
##### Data Preprocessing & Cleaning - Pandas, numpy
  - Handle Missing Data
  - Outlier Detection & Removal
  - Feature Engineering
##### Dimentionsnality reduction - SVD, PCA from sklearn
##### Model building - various regression/classification techniques.
- Modeling Algorithms:
   - Logistic Regression
   - Random Forest
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
   - Decision Tree
   - Naive Bayes

- Evaluation Metrics: Accuracy, F1 Score, ROC-AUC, Confusion Matrix
- Model Selection: GridSearchCV with StratifiedKFold

#### Results
For now, based on the baseline model KNN results, below is the summary. The KNN model achieved strong results:
- Accuracy: 97.38%
- Recall (Sensitivity): 93.46%
- Precision: 98.87%
- F1 Score: 96.09%
- ROC AUC Score: 98.95%

- Among the evaluation metrics, Recall (Sensitivity) stands out as the most important, achieving 93.46%, which indicates that the model successfully identifies the vast majority of diabetic patients, a critical requirement in healthcare settings where missing true cases can have serious consequences.
- Additionally, the model achieves a high F1 Score of 96.09%, reflecting a strong balance between precision and recall, while the ROC AUC Score of 98.95% confirms its excellent ability to distinguish between diabetic and non-diabetic individuals.
- With a Precision of 98.87% and overall Accuracy of 97.38%, the model also minimizes false positives and performs reliably across the dataset. 

#### Next steps
- Evaluate additional models (e.g., Random Forest, Logistic Regression, XGBoost) for comparison.
- Use cross-validation and hyperparameter tuning to improve robustness.
- Integrate clinical domain knowledge to improve feature selection.
- Deploy the model in a web app or clinical decision support interface.
- Expand dataset to include more diverse demographics for generalizability.

                                                   
#### Outline of project

- https://github.com/RamyaChowdavaram/early-diabetes-detection-system/blob/main/diabetes_model.ipynb

##### Ramya Chowdavaram

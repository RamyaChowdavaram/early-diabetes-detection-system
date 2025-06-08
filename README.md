# Early Diabetes Detection System (EDDS)
## An overview of the question to be solved
Diabetes is a chronic medical condition that affects millions of people worldwide. Early detection and proper management of diabetes can help prevent severe complications such as heart disease, kidney failure, and nerve damage. The goal of this project is to develop a machine learning model that can predict whether a person is likely to develop diabetes based on their medical and demographic data.
How can this model assist doctors and healthcare professionals, common people in early diabetes detection? and how can this model be integrated into a web or mobile application for public use?

## Identification of the type of data that will be used to solve the question.
The data should include various health parameters, lifestyle habits, and genetic predispositions that contribute to diabetes risk. The data should be structured with realistic distributions.
The dataset should be useful for exploring the relationships between lifestyle choices, genetic factors, and diabetes risk, providing valuable insights for predictive modeling and health analytics.
The data should include some or most of the below features:

-  Age: The age of the individual (18-90 years).
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

## List of 1–3 techniques that could be used to answer the question identified.
### Explore & Understand the Data (Exploratory Data Analysis) - Pandas, numpy, matlib, plotly, seaborn
  - Understand the types of variables (numerical, categorical).
  - Explore the shape of the data (rows and columns).
  - Check for missing values, duplicates, or incorrect data types.
  - **Visualize** the data to identify trends, outliers, and correlations between features.
  - Check basic statistics like mean, median, standard deviation, and distribution of data.
  - Use histograms, box plots, and pair plots to visualize data relationships.
### Data Preprocessing & Cleaning - Pandas, numpy
  - Handle Missing Data
  - Outlier Detection & Removal
  - Feature Engineering
### Dimentionsnality reduction - SVD, PCA from sklearn
### Model building - various regression/classification techniques.


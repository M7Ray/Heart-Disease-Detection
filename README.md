# Heart-Disease-Detection

PROJECT OVERVIEW

Early detection and prediction of heart failure can help healthcare professionals to intervene before the condition worsens and lead to severe complications. The objective of this project is to analyze a heart failure dataset and build a machine learning model to predict the likelihood of heart failure based on patient demographics, clinical and lab measurements, and other risk factors. The motivation for this project is to develop a reliable and accurate tool that can assist healthcare professionals in identifying patients who are at high risk of heart failure and provide timely interventions.

The data set contains the following parameters:
Categorical Variables: sex, ChestPainType, and fasting blood sugar
Numerical Variables: Age, Cholesterol, and blood pressure
Others:  RestingBP, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease

INSIGHTS 

Methods used: 
- Exloratory Data Analysis: Random Forest Classifier, Decision Tree Classifier, Naive Bayes Classifier, ANN
- Data Cleaning: remove duplicates, handle missing values, removing unnecessary columns.
- Data Visualization
- Decision Tree: splitting the dataset into training and testing sets, and used various classification algorithms.
- Model Evaluation: Accuracy, Precision, Recall and F1 score.

RESULTS

Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak and ST_Slope variables were strongly correlated with heart failure, and the correlation was more pronounced in patients with a history of heart disease. After training and testing the models, the Decision Tree Classifier achieved the Accuracy score of 0.82, Precision score of 0.92, Recall score of 0.75 and F1 score of 0.82. While using the Random Forest Classifier has an accuracy of 0.88, precision of 0.92, recall of 0.87, F1 score of 0.89. For the Naive Bayes Classifier, the accuracy was 0.83, precision 0.86, Recall 0.85, F1 score 0.85. Lastly, using the ANN classifier, the obtained accuracy was 0.83, precision 0.88, Recall and F1 are 0.85.

CONCLUSION

The project demonstrated the use of machine learning algorithms to predict the likelihood of heart failure based on patient demographics, clinical and lab measurements, and other risk factors. The results showed that Age, Cholesterol, MaxHR and RestingBP levels were the most significant predictors of heart failure, and the Decision Tree Classifier achieved a high accuracy in predicting heart failure. These findings could potentially assist healthcare professionals in identifying patients who are at high risk of heart failure and provide timely interventions.

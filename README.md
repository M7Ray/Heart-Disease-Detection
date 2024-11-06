# Heart-Disease-Detection

- PROJECT OVERVIEW

    Early detection and prediction of heart failure can help healthcare professionals to intervene before the condition worsens and lead to severe complications. The           objective of this project is to analyze a heart failure dataset and build a machine learning model to predict the likelihood of heart failure based on patient 
    demographics, clinical and lab measurements, and other risk factors. The motivation for this project is to develop a reliable and accurate tool that can assist 
    healthcare professionals in identifying patients who are at high risk of heart failure and provide timely interventions.

    The data set contains the following parameters:
    Categorical Variables: sex, ChestPainType, and fasting blood sugar
    Numerical Variables: Age, Cholesterol, and blood pressure
    Others:  RestingBP, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease

- INSIGHTS 

    Methods used: 
    1- Exloratory Data Analysis: Random Forest Classifier, Decision Tree Classifier, Naive Bayes Classifier, ANN
    
    2- Data Cleaning: remove duplicates, handle missing values, removing unnecessary columns.
    
    3- Data Visualization
    
    4- Decision Tree: splitting the dataset into training and testing sets, and used various classification algorithms.
    
    5- Model Evaluation: Accuracy, Precision, Recall and F1 score.

- RESULTS

    Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak and ST_Slope variables were strongly correlated with heart 
    failure, and the correlation was more pronounced in patients with a history of heart disease. After training and testing the models, the Decision Tree Classifier 
    achieved the Accuracy score of 0.82, Precision score of 0.92, Recall score of 0.75 and F1 score of 0.82. While using the Random Forest Classifier has an accuracy of 
    0.88, precision of 0.92, recall of 0.87, F1 score of 0.89. For the Naive Bayes Classifier, the accuracy was 0.83, precision 0.86, Recall 0.85, F1 score 0.85. Lastly, 
    using the ANN classifier, the obtained accuracy was 0.83, precision 0.88, Recall and F1 are 0.85.

- CONCLUSION
    
    The project demonstrated the use of machine learning algorithms to predict the likelihood of heart failure based on patient demographics, clinical and lab 
    measurements, and other risk factors. The results showed that Age, Cholesterol, MaxHR and RestingBP levels were the most significant predictors of heart failure, and 
    the Decision Tree Classifier achieved a high accuracy in predicting heart failure. These findings could potentially assist healthcare professionals in identifying 
    patients who are at high risk of heart failure and provide timely interventions.

Dataset Link
Markup :  [Named Link](http://www.google.fr/ "Named link title") and http://www.google.fr/ or <http://example.com/>

# Reading and Cleaning The Data

```python

import numpy as np 
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

datapath = pd.read_csv("Heart Failure Data.csv")
datapath

#check if there is missing data.
datapath = datapath.replace([np.inf, -np.inf], np.nan)
# Drop any rows with NaN values
datapath = datapath.dropna()

#printing number of missing data in each column.
datapath.isna().sum()

```
<img width="554" alt="t1" src="https://github.com/user-attachments/assets/83db2e4b-bfc4-41f8-8185-5cc688330a22">
<img width="98" alt="t2" src="https://github.com/user-attachments/assets/b94e692c-f84c-4d51-a64c-6bd98e76d24c">

# Preprocessing the data

```python

# Preprocessing the data by convert categorical variables into numerical variables. 
datapath["Sex"] = datapath["Sex"].replace({"M": 0, "F": 1})
datapath["ExerciseAngina"] = datapath["ExerciseAngina"].replace({"N": 0, "Y": 1})
datapath["ChestPainType"] = datapath["ChestPainType"].replace({"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3})
datapath["RestingECG"] = datapath["RestingECG"].replace({"Normal": 0, "ST": 1, "LVH": 2 })
datapath["ST_Slope"] = datapath["ST_Slope"].replace({"Up": 0, "Flat": 1, "Down": 2 })
datapath
datapath.describe().style.background_gradient(cmap='Greens')
cat_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'HeartDisease']

# Create a 2x4 grid of subplots
fig, axs = plt.subplots(2,4,figsize=(25, 15))

# Flatten the grid of subplots to simplify indexing
axs = axs.flatten()

# Plot each categorical column on a specific subplot
for i, col in enumerate(cat_cols[0:7]):
    sns.countplot(x=col, hue='HeartDisease', data=datapath, ax=axs[i])
    axs[i].set_title(f"{col} Distribution by Heart Disease")
    axs[i].set_xlabel(col)
    axs[i].set_ylabel("Count")
    for p in axs[i].patches:
        height = p.get_height()
        axs[i].text(p.get_x()+p.get_width()/2., height+3, f"{height}", ha="center")

# Display the plots
plt.show()

sns.pairplot(data=datapath,hue='HeartDisease')

```

<img width="554" alt="t3" src="https://github.com/user-attachments/assets/493c4c4e-f4d2-4584-823b-8e18613d4488">
<img width="585" alt="t4" src="https://github.com/user-attachments/assets/e44e7606-f3cb-4d22-91d2-0126e3888cce">
<img width="599" alt="t5" src="https://github.com/user-attachments/assets/4f1981d3-207b-4a50-adba-102848a6005b">
<img width="600" alt="t6" src="https://github.com/user-attachments/assets/b3a3fefd-acf4-4bbf-9631-374a09c9fc28">



```python
num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

for col in num_cols:
    fig = px.histogram(datapath, x=col, color='HeartDisease')
    fig.update_layout(title=f"{col} Distribution by Heart Disease", xaxis_title=col, yaxis_title="Count")
    fig.show()

correlation  = datapath.corr()
correlation

```

<img width="593" alt="t7" src="https://github.com/user-attachments/assets/c43813d2-e73f-4ccf-8c0c-6befeb135c0f">


```python

plt.figure(figsize=(15,10))
plt.subplot(2,3,1)
sns.boxplot(data=datapath,x='HeartDisease',y='Age',hue='Sex')
plt.title('HeartDisease over Age')
plt.subplot(2,3,2)
sns.boxplot(data=datapath,x='HeartDisease',y='RestingBP',hue='Sex')
plt.title('HeartDisease over RestingBP')
plt.subplot(2,3,3)
sns.boxplot(data=datapath,x='HeartDisease',y='Cholesterol',hue='Sex')
plt.title('HeartDisease over Cholesterol')
plt.subplot(2,3,4)
sns.boxplot(data=datapath,x='HeartDisease',y='MaxHR',hue='Sex')
plt.title('HeartDisease over MaxHR')
plt.subplot(2,3,5)
sns.boxplot(data=datapath,x='HeartDisease',y='Oldpeak',hue='Sex')
plt.title('HeartDisease over Oldpeak')
plt.tight_layout()

```

<img width="593" alt="t8" src="https://github.com/user-attachments/assets/0ab27b92-2bf4-4534-8635-f93cb9e321d9">



```python

fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(correlation, annot=True, fmt='0.2f', cmap="viridis", ax=ax)

```

<img width="361" alt="t9" src="https://github.com/user-attachments/assets/10ac26ad-2745-4db3-a91f-c57cffaf1057">


```python


import matplotlib.pyplot as plt

num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
num_cols_len = len(num_cols)

fig, axs = plt.subplots(1, num_cols_len, figsize=(25,5))

for i in range(num_cols_len):
    col = num_cols[i]
    ax = axs[i]
    sns.histplot(data=datapath, x=col, hue='HeartDisease', ax=ax, kde=True)
    ax.set_title(f"{col} Distribution by Heart Disease")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.legend(['No Heart Disease', 'Heart Disease'])

plt.show()

```

<img width="599" alt="t10" src="https://github.com/user-attachments/assets/7b71c3ea-16b6-4238-a0cf-8f82a691e68c">


```python


X = datapath.drop("HeartDisease", axis=1)
y = datapath["HeartDisease"]

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=69)
model_1 = RandomForestClassifier()
model_2 = DecisionTreeClassifier()
model_3 = GaussianNB()
model_4 = MLPClassifier()



print("Number of Trained Data :",X_train1.shape)
print("Number of Tested Data: ",X_test1.shape)

print("-----------------------------------------------------------")

model_1.fit(X_train1, y_train1)
y_pred = model_1.predict(X_test1)
print("# Using Random Forest Classifier: ")
print("Accuracy:", accuracy_score(y_test1, y_pred))
print("Precision:", precision_score(y_test1, y_pred))
print("Recall:", recall_score(y_test1, y_pred))
print("F1 score:", f1_score(y_test1, y_pred))
RF_test_score = round((accuracy_score(y_test1, y_pred)) * 100, 2)

print("-----------------------------------------------------------")

model_2.fit(X_train1, y_train1)
y_pred = model_2.predict(X_test1)
print("# Using Decision Tree Classifier: ")
print("Accuracy:", accuracy_score(y_test1, y_pred))
print("Precision:", precision_score(y_test1, y_pred))
print("Recall:", recall_score(y_test1, y_pred))
print("F1 score:", f1_score(y_test1, y_pred))
DT_test_score = round((accuracy_score(y_test1, y_pred)) * 100, 2)


print("-----------------------------------------------------------")

model_3.fit(X_train1, y_train1)
y_pred = model_3.predict(X_test1)
print("# Using Naive Bayes Classifier: ")
print("Accuracy:", accuracy_score(y_test1, y_pred))
print("Precision:", precision_score(y_test1, y_pred))
print("Recall:", recall_score(y_test1, y_pred))
print("F1 score:", f1_score(y_test1, y_pred))
NB_test_score = round((accuracy_score(y_test1, y_pred)) * 100, 2)

print("-----------------------------------------------------------")

model_4.fit(X_train1, y_train1)
y_pred = model_4.predict(X_test1)
print("# Using Artificial Neural Network (ANN) Classifier: ")
print("Accuracy:", accuracy_score(y_test1, y_pred))
print("Precision:", precision_score(y_test1, y_pred))
print("Recall:", recall_score(y_test1, y_pred))
print("F1 score:", f1_score(y_test1, y_pred))
ANN_test_score = round((accuracy_score(y_test1, y_pred)) * 100, 2)
print("-----------------------------------------------------------")


```

<img width="249" alt="t11" src="https://github.com/user-attachments/assets/87620b93-94ad-4237-b301-cb2c565208c3">


# Model Evaluation


```python


models = pd.DataFrame({
    'Model': ['Random Forest','Decision Tree','Naive Bayes','Artificial Neural Network'],
    'Score': [ RF_test_score , DT_test_score , NB_test_score , ANN_test_score]})
models.sort_values(by='Score', ascending=False).style.background_gradient(cmap='Greens')


```

<img width="200" alt="t12" src="https://github.com/user-attachments/assets/b36829cd-a858-4071-8e49-99de0e362b73">


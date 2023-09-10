---------- Import packages ----------

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# metrics and helpful functions
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree

import pickle


---------- EDA & Data Cleaning ----------

# Load dataset into a dataframe
df0 = pd.read_csv("HR_capstone_dataset.csv")

# Display first few rows of the dataframe
df0.head()
df0.describe()
df0.info()

# Fix column names
df0.rename(columns={'Work_accident': 'work_accident',
                          'average_montly_hours': 'average_monthly_hours',
                          'time_spend_company': 'tenure',
                          'Department': 'department'}, inplace=True)

# Check for missing values
df0.isna().any()

# Check for duplicates
df0.duplicated().sum()

# Inspect duplicates and removing them
df_dup = df0[df0.duplicated()]
df_dup.head(20)
df1 = df0.drop_duplicates()

# Create a boxplot of `tenure` and detect outliers
fig = plt.figure(figsize = (12, 2))
sns.boxplot(df1['tenure'])
plt.title('Tenure Boxplot', fontsize=12)

# Get numbers of people who left vs. stayed
print('Number of people left: ', df1['left'].sum())
print('Number of people stayed: ', len(df1) - df1['left'].sum())
# Get percentages of people who left vs. stayed
print('Percent left: ', round(100*(df1['left'].sum()/len(df1)), 2), '%')
print('Percent stayed: ', round(100*((len(df1) - df1['left'].sum())/len(df1)), 2), '%')


---------- Data Visualization ----------
# Histograms of left/stayed by Numbers of projects and tenure
fig, ax = plt.subplots(1, 2, figsize = (21, 7))
sns.histplot(data=df1, x='number_project', hue='left', multiple='dodge', shrink=2, ax=ax[0])
sns.histplot(data=df1, x='tenure', hue='left', multiple='dodge', shrink=6, ax=ax[1])
ax[0].set_title('Number of Project Histogram', fontsize='16')
ax[1].set_title('Tenure Histogram', fontsize='16')

# Histogram of left/stayed by department
fig = plt.figure(figsize = (10, 6))
sns.histplot(data=df1, x='department', hue='left', multiple='dodge', shrink=.5)
plt.xticks(rotation='45')

# Histogram of salary by Tenure
fig = plt.figure(figsize = (10, 6))
sns.histplot(data=df1, x='tenure', hue='salary', multiple='dodge', shrink=5)

# Create Boxplot of satisfaction level by tenure
fig = plt.figure(figsize = (9, 7))
bx = sns.boxplot(data=df1, 
                 x='satisfaction_level', 
                 y='tenure', hue='left', 
                 orient='h', 
                 linewidth=1.5,
                 flierprops={"marker": "o", "markersize": 3})
bx.set_title('Satisfaction Level By Tenure')
bx.invert_yaxis()

#Scatterplot of satisfaction level by average monthly hours
plt.figure(figsize=(16, 9))
sns.scatterplot(df1['average_monthly_hours'], df1['satisfaction_level'], hue=df1['left'], alpha = 0.5)
plt.legend(labels=['left', 'stayed'])

# correlation heatmap
plt.figure(figsize=(16, 9))
heatmap = sns.heatmap(df0.corr(), 
                      vmin=-1, vmax=1, 
                      annot=True, 
                      cmap=sns.color_palette("coolwarm", as_cmap=True), 
                      linewidth = 0.1)


---------- Building Logistic Regression Model ----------

df_code = df1.copy()

# Encode the `salary` column as an ordinal numeric category
df_code['salary'] = (
    df_code['salary'].astype('category')
    .cat.set_categories(['low', 'medium', 'high'])
    .cat.codes
)

# Dummy encode the `department` column
df_code = pd.get_dummies(df_code, drop_first=False)

# Excluding tenure outliers
df_final = df_code[(df_code['tenure'] >= 2) & (df_code['tenure'] <= 5)]

# Splitting the data
y = df_final['left']
X = df_final.drop('left', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Train, predict and make confusion matrix
log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)
y_pred = log_clf.predict(X_test)

logreg_matrix = confusion_matrix(y_test, y_pred, labels=log_clf.classes_)
matrix_display = ConfusionMatrixDisplay(confusion_matrix=logreg_matrix, display_labels=log_clf.classes_)
matrix_display.plot(values_format='')
plt.show()

print(classification_report(y_test, y_pred, 
                            target_names = ['Predicted would not leave', 'Predicted would leave']))


---------- Decision Tree Model ----------

# Implementing GridSearchCV to find best model parameters
tr = DecisionTreeClassifier(random_state=0)

params = {
    'max_depth':range(1, 8),
    'min_samples_leaf':[2, 5, 1],
    'min_samples_split':range(2, 6)
}
scores = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

tr_grid = GridSearchCV(tr, params, scoring=scores, refit='roc_auc')
tr_grid.fit(X_train, y_train)
print(tr_grid.best_params_)
print(tr_grid.best_score_)

# Plot Tree
plt.figure(figsize=(40,15))
plot_tree(tr_grid.best_estimator_, max_depth=4, fontsize=14, feature_names=X.columns, 
          class_names={0:'stayed', 1:'left'}, filled=True);
plt.show()

# Plot Feature importance
tr_importances = pd.DataFrame(tr_grid.best_estimator_.feature_importances_, 
                                 columns=['gini'], 
                                 index=X.columns
                                )
tr_importances = tr_importances.sort_values(by='gini', ascending=False)
tr_importances = tr_importances[tr_importances['gini'] != 0]
tr_importances

sns.barplot(data=tr_importances, x="gini", y=tr_importances.index, orient='h')
plt.title("Decision Tree: Feature Importances for Employee Leaving", fontsize=12)
plt.ylabel("Feature")
plt.xlabel("Importance")
plt.show()

---------- Feature Engineering & Random Forest Model ----------

# Define 'overworked' as > 175 hrs/week
df_final['overworked'] = df_final['average_monthly_hours']

print('Max hours:', df_final['overworked'].max())
print('Min hours:', df_final['overworked'].min())

df_final['overworked'] = (df_final['overworked'] > 175).astype(int)
df_final.drop('average_monthly_hours', axis=1)

# Splitting data
y = df_final['left']
X = df_final.drop('left', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

# Train, Grid Search
rand_f = RandomForestClassifier(random_state=0)
cv_params = {'max_depth': [3,5, None], 
             'max_features': [1.0],
             'max_samples': [0.7, 1.0],
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'n_estimators': [300, 500],
             }  
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

rand_grid = GridSearchCV(rand_f, cv_params, scoring=scoring, cv=4, refit='roc_auc')
rand_grid.fit(X_train, y_train)
print(rand_grid.best_params_)
print(rand_grid.best_score_)

# Predict and make confusion matrix
preds = rand_grid.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, preds, labels=rand_grid.classes_)
# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=rand_grid.classes_)
disp.plot(values_format='')

---------- End of Code ----------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.show()
import scipy.stats as stats
from scipy.stats import skew, norm, probplot, boxcox, f_oneway
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("diabetes.csv")

# df['Outcome'] = df['Outcome'].astype('category')

# plt.figure(figsize=(20,30))

# for i, variable in enumerate(df):
#                      plt.subplot(5,4,i+1)
#                      plt.boxplot(df[variable],whis=1.5)
#                      plt.tight_layout()
#                      plt.title(variable)
                    
# plt.show()

# # Use flooring and capping method
# def treat_outliers(df,col):
#     Q1=df[col].quantile(0.25) 
#     Q3=df[col].quantile(0.75) 
#     IQR=Q3-Q1
#     Lower_Whisker = Q1 - 1.5*IQR 
#     Upper_Whisker = Q3 + 1.5*IQR
#     df[col] = np.clip(df[col], Lower_Whisker, Upper_Whisker)                                                            
#     return df

# def treat_outliers_all(df, col_list):
#     for c in col_list:
#         df = treat_outliers(df,c)
#     return df

# numerical_col = df.select_dtypes(include=np.number).columns.tolist()
# df = treat_outliers_all(df,numerical_col)
# plt.figure(figsize=(20,30))

# for i, variable in enumerate(numerical_col):
#                      plt.subplot(5,4,i+1)
#                      plt.boxplot(df[variable],whis=1.5)
#                      plt.tight_layout()
#                      plt.title(variable)

# plt.show()

from sklearn.model_selection import train_test_split
import scipy.stats as stats
from scipy.stats import stats, norm, skew
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import linear_model
from sklearn import metrics
from sklearn import datasets

data=df.copy()
data.info()

X = data.drop('Outcome',axis=1)    # Features
y = data['Outcome'].astype('int64') # Labels (Target or Outcome Variable)
# converting target to integers - since some functions might not work with bool type
# Splitting data into training and test set:
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3, random_state=1)
print(X_train.shape, X_test.shape)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='gini',class_weight={0:0.15,1:0.85},random_state=1)

# 2 commonly used splitting criteria are Gini impurity and information gain (entropy)
# Gini: measures the probability of misclassifying a randomly chosen element if it were randomly labeled
    # Would goal be to minimize or maximize the Gini impurity when making splits???
        # MINIMIZE
    
    
# Information Gain (Entropy): entropy measures impurity or uncertainty, while information gain quantifies reduction in entropy
    # Which do we want to minimize? Maximize?
        # MINIMIZE Entropy
        # MAXIMIZE Information Gain

model.fit(X_train, y_train)

def make_confusion_matrix(model,y_actual,labels=[1, 0]):
    y_predict = model.predict(X_test)
    cm=metrics.confusion_matrix( y_actual, y_predict, labels=[0, 1])
    df_cm = pd.DataFrame(cm, index = [i for i in ["Actual - No","Actual - Yes"]],
                  columns = [i for i in ['Predicted - No','Predicted - Yes']])
    group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in
              zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=labels,fmt='')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
make_confusion_matrix(model,y_test)


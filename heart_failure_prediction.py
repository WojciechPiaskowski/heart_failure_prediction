# imports
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# style options
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 30)
sns.set_style('whitegrid')

# load in the data
df = pd.read_csv('heart_failure_clinical_records_dataset.csv', sep=',', header=0)

# first look at the data (dataset from kaggle)
df.head()
df.describe()
print('number of columns: ', len(df.columns))
# 299 observations, 12 exogenic variables (features) and 1 endogenic variable - the even of death (1 - 0 dummy variable)

# data exploration

sns.countplot(x='DEATH_EVENT', data=df)
# data is not balanced, there is a lot more observations with non-death patients compared to patients that died
# that means accuracy might not be the best metric, and using F1 score as balanced metric of recall and precission
# might better

sns.distplot(df['age'], bins=15, color='red')
# distribution of patients age is similar to normal distribution, but it start with patients aged 40 and end with
# patients aged 90. That means the predictions of the model should only be made on people in this spectrum

df.corr()['DEATH_EVENT'].sort_values(ascending=False)
# the variables with a strongest correlation to death_event are time (which is just an index, ejection_fraction,
# serum_creatinine, age
# the weakest correlation is with variables sex and diabetes
# however since Y is discrete, it could be good to do some other measures of correlation like Chi-squared
# also correlation does not necessarily mean causation

# lets plot ejection_fraction variable
sns.boxplot(y='ejection_fraction', x='high_blood_pressure', data=df, hue='sex')
sns.boxplot(y='ejection_fraction', x='sex', data=df, hue='DEATH_EVENT')
# it seems that the probability of death is higher when the ejection_fraction is lower
# is is also generally a bit lower chance of death for women as opposed to men

sns.distplot(df['age'], bins=15, color='red')
sns.distplot(df['age'][df['DEATH_EVENT'] == 1])

df['age'][df['DEATH_EVENT'] == 0].describe()
df['age'][df['DEATH_EVENT'] == 1].describe()
# people who end up dying are on average 65 years old (with similar median) and standard deviation of 13.2,
# while people who survive are on average 58.7 (median 60) with a standard deviation of 10
# age does not seem like a very strong factor, but increase the chance of death to some extent

# possible more task for data exploration: take a closer look at distribution of each variable, its quantiles,
# mean, standard deviation or variance, coefficient of variation

# lets cast the data into numpy arrays
X = np.array(df.iloc[:, 0:11])
Y = np.array(df['DEATH_EVENT'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=52)

# let try logistic regression using stats models to see which exogenic variables are statistically significant
# (using chi-squared / likelyhood tests)

model1 = sm.Logit(endog=Y_train, exog=sm.add_constant(X_train))
r1 = model1.fit()
print(r1.summary())

# it looks like variables x2, x3, x4, x6, x7, x9, x10, x11 as well as constant
# - if we take alpha = 0.05 as the significance level

# here we could do a lot inference as well as likelyhood tests related to dropping variables, compare models with AIC
# etc. but instead we will see the results, then try to drop those variables and see results again

def results(y, yhat):
    print(confusion_matrix(y,yhat), '\n', classification_report(y, yhat))
# simple function to see results of predictions

# lets try the current model
yhat_prob = model1.predict(r1.params, exog=sm.add_constant(X_test))
yhat = (yhat_prob > 0.5).astype(int)

results(Y_test, yhat)
# results are fairly bad for predicting death event, while reasonably okay for predicting the cases when the patient
# did not die -- overall F1-score at 0.70
# here we could do additional measures like AUC, draw a ROC curve, GINI index

# lets try removing insignificant variables and the constant
X = np.array(df.drop(['anaemia', 'creatinine_phosphokinase', 'diabetes', 'high_blood_pressure', 'platelets',
                      'serum_sodium', 'sex', 'smoking', 'time', 'DEATH_EVENT'], axis=1))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=52)

model2 = LogisticRegression()
model2.fit(X_train, Y_train)
yhat = model2.predict(X_test)

results(Y_test, yhat)

# the F1-score got a bit higher - 0.76

# again, before going further we could do some more statistical inference, but lets focus on improving that F1-score
# lets try different methods of estimation with current variables
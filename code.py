# --------------
# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# Code starts here
# load the dataset using pandas read_json
df = pd.read_json(path, lines=True)

# analyse the dataset
print(df.shape)
print(df.columns)
print(df.info)
print(df.describe())

# replace the space in column names with "_"
df.columns = df.columns.str.strip().str.replace(" ","_")

# check for the missing values 
missing_data = pd.DataFrame({'Total_Missing':  df.isna().sum(),'Percentage_missing':df.isna().sum() *100/df.shape[0]})

print(missing_data)

# on analysing the missing data we see that bust,shoe_size, shoe_width, waist have high percentage of 
# missing values , also user_name, review_text and review summary is irrelevant
df.drop(columns=['waist', 'bust', 'user_name','review_text','review_summary','shoe_size','shoe_width'], axis=1, inplace=True)

# split the data into features and labels
X = df.drop(columns=['fit'], axis=1)
y = df['fit']

# fit the features and labels into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6, test_size=0.33)

# check the shape of splitted data
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Code ends here


# --------------
def plot_barh(df,col, cmap = None, stacked=False, norm = None):
    df.plot(kind='barh', colormap=cmap, stacked=stacked)
    fig = plt.gcf()
    fig.set_size_inches(24,12)
    plt.title("Category vs {}-feedback -  cloth {}".format(col, '(Normalized)' if norm else ''), fontsize= 20)
    plt.ylabel('Category', fontsize = 18)
    plot = plt.xlabel('Frequency', fontsize=18)


# Code starts here

#groupby category feature
g_by_category = df.groupby(['category'])

# check the value counts based on fit
cat_fit = g_by_category['fit'].value_counts(normalize=True)

# unstack
cat_fit = cat_fit.unstack()

# plot the horozontal bar chart
plot_barh(cat_fit, cat_fit.columns)

# Code ends here


# --------------
# Code starts here

#value counts of g_by_category based on length
cat_len = g_by_category['length'].value_counts()

#unstack the columns
cat_len = cat_len.unstack()

#plot horizontal bar
plot_barh(cat_len, cat_len.columns)

# Code ends here


# --------------
# Code starts here

# function to convert feet to cms
def get_cms(x):
    try:
        if (type(x)== str) and (len(x) > 1):
            return int(x[0]) * 30.48 + int(x[4:-2])*2.54
    except:
        res = 'Invalid Input'


# apply the function to convert feet to cms and leave the NaN's as it is
X_train.height = X_train.height.apply(get_cms)
X_test.height = X_test.height.apply(get_cms)

# verify the conversion
print(X_train.height.head())
print(X_test.height.head())
# Code ends here


# --------------
# Code starts here
from sklearn.preprocessing import Imputer

# check missing values for X_train
print(X_train.isna().sum())

# drop the NaN values from rows of ['height','length','quality'] as they have limited missing values
X_train.dropna(axis=0, how='any', subset=['height','length','quality'], inplace=True)
X_test.dropna(axis=0 , how='any', subset=['height','length','quality'], inplace= True)

# removing indexes from y_train and y_test which are not present in X_train and X_test
y_train =  y_train[y_train.index.isin(idx for idx in X_train.index)]
y_test = y_test[y_test.index.isin(idx for idx in X_test.index)]

# now we'll work on imputing the missing values for numerical columns['bra_size','hips']

# initialising Imputer 
mean_imputer = Imputer(strategy = 'mean')

# fit and transform the train data for bra size
X_train[['bra_size']] = mean_imputer.fit_transform(X_train[['bra_size']])
# transform the test data for bra size
X_test[['bra_size']] = mean_imputer.transform(X_test[['bra_size']])

# fit and transform the train and test data for hips
X_train[['hips']] = mean_imputer.fit_transform(X_train[['hips']])
X_test[['hips']] = mean_imputer.transform(X_test[['hips']])

# we now need to impute the missing values for categorical columns, by replacing them with mode
# calculate mode for cup size for train and test set and replace the missing with them

mode_1 = X_train['cup_size'].mode()[0]
mode_2 = X_test['cup_size'].mode()[0]

# replace the na values
X_train['cup_size'].fillna(value= mode_1, inplace= True)
X_test['cup_size'].fillna(value= mode_2 , inplace= True)

# Verify again for the presence of missing value
print(X_train.isna().sum())
print(X_test.isna().sum())



# Code ends here


# --------------
# Code starts here

#converting the categorical values to one hot encoding using pandas dummy function

X_train = pd.get_dummies(X_train, columns=['category','cup_size','length'])

X_test = pd.get_dummies(X_test , columns = ['category','cup_size','length'])

# verifying the encoding
print(X_train.head())
print(X_test.head())
# Code ends here


# --------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score


# Code starts here

# initialise the dicision tree classifier
model = DecisionTreeClassifier(random_state=6)

# fit the model to training data
model.fit(X_train , y_train)

# predict the output
y_pred = model.predict(X_test)

# calculate the accuracy score
score = accuracy_score(y_test, y_pred)
print('Accuracy score of the Decision tree classifier is:', score)

# calculate the precision
precision = precision_score(y_test, y_pred, average = None)
print('Precision of the model is:', precision)
# has high precision value for fit , while low precision values for small and large

# Code ends here


# --------------
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


# parameters for grid search
parameters = {'max_depth':[5,10],'criterion':['gini','entropy'],'min_samples_leaf':[0.5,1]}

#initialise the decision tree model
model = DecisionTreeClassifier(random_state=6)

# initialise the gridsearchCV
grid = GridSearchCV(estimator= model , param_grid = parameters)

# fit the model to train data
grid.fit(X_train, y_train)

# predit the label using the gridsearchcv model
y_pred = grid.predict(X_test)

# calculate the accuracy score
accuracy = grid.score(X_test, y_test)

print('Accuracy of the Decision Tree Model after prunning the tree is:', accuracy)


# Code ends here



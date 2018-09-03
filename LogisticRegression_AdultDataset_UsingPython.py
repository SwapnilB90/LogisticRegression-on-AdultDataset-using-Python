import numpy as np
import pandas as pd

adult_df=pd.read_csv(r'F:\Imarticus\Python\Datasets\adult_data.csv',header=None,
                      delimiter=' *, *',engine='python')
adult_df.head()
adult_df.shape
adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
'marital_status', 'occupation', 'relationship',
'race', 'sex', 'capital_gain', 'capital_loss',
'hours_per_week', 'native_country', 'income']
adult_df.head()
adult_df.isnull().sum()  #for numerical missing values

#%%
#for categorical missing values
for value in ['workclass', 'education',
            'marital_status', 'occupation', 'relationship',
            'race', 'sex','native_country', 'income']:
     print(value,':',sum(adult_df[value]=='?'))
 
#%%
 #Create copy of dataframe
adult_df_rev=pd.DataFrame.copy(adult_df)
adult_df_rev.describe(include='all')
 #%%
 #replace missing values with mode
for value in ['workclass', 'occupation', 'native_country']:
     adult_df_rev.replace(['?'],[adult_df_rev[value].mode()[0]],inplace=True)
adult_df_rev.head(20) 

 #%%
for value in ['workclass', 'education',
            'marital_status', 'occupation', 'relationship',
            'race', 'sex','native_country', 'income']:
     print(value,':',sum(adult_df_rev[value]=='?'))
#%%     
#creating a list of categorical variables
colname = ['workclass', 'education',
            'marital_status', 'occupation', 'relationship',
            'race', 'sex','native_country', 'income']
print(colname)
#%%
#For preprocessing the data(for converting categorical data to numerical)
from sklearn import preprocessing
le ={}
for x in colname:
    le[x]=preprocessing.LabelEncoder()
    
for x in colname:
    adult_df_rev[x]=le[x].fit_transform(adult_df_rev.__getattr__(x))

print(adult_df_rev.head(20))

#0=> <50K
#1=> >50K

#%%
#segregetting dependent & independent variables
x=adult_df_rev.values[:,:-1]
y=adult_df_rev.values[:,-1]
print(y)
#%%
#For scaling the data to normalize fashion
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

scaler.fit(x)
x=scaler.transform(x)
print(x)
#%%
y=y.astype(int)
#%%
from sklearn.model_selection import train_test_split

#split the data into test and train
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)
#%%
#running the model
from sklearn.linear_model import LogisticRegression
classifier=(LogisticRegression())
#fitting training data to model
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
print(list(zip(y_test,y_pred)))
print(y_pred)
#%%
#Evaluating the model
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(y_test,y_pred)
print(cfm)
print('classification report:')
print(classification_report(y_test,y_pred))
accuracy_score=accuracy_score(y_test,y_pred)
print('accuracy of model:',accuracy_score)
#%%
#Store the predicted probabilities
y_pred_prob = classifier.predict_proba(x_test)
print(y_pred_prob)
#%%
y_pred_class=[]
for value in y_pred_prob[:,0]:
    if value < 0.77:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
print(y_pred_class)
#%%
from sklearn.metrics import confusion_matrix,accuracy_score
cfm=confusion_matrix(y_test.tolist(),y_pred_class)
print(cfm)
accuracy_score=accuracy_score(y_test.tolist(),y_pred_class)
print('Accuracy of model:',accuracy_score)
#%%
#using cross validation
classifier=(LogisticRegression())
from sklearn import cross_validation
#performing kfold cross_validation
kfold_cv=cross_validation.KFold(n=len(x_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=x_train,y=y_train,
                                                 scoring='accuracy',cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())

for train_value, test_value in kfold_cv:
    classifier.fit(x_train[train_value],y_train[train_value]). \
    predict(x_train[test_value])
y_pred=classifier.predict(x_test)
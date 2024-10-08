# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```python
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![alt text](<Screenshot 2024-10-08 111034.png>)

```python
df.dropna()
```
![alt text](<Screenshot 2024-10-08 111041.png>)

```python
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![alt text](<Screenshot 2024-10-08 111046.png>)

```python
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![alt text](<Screenshot 2024-10-08 111053.png>)

```python
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![alt text](<Screenshot 2024-10-08 111135.png>)

```python
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![alt text](<Screenshot 2024-10-08 111140.png>)

```python
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![alt text](<Screenshot 2024-10-08 111147.png>)

```python
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```
![alt text](<Screenshot 2024-10-08 111152.png>)

## FEATURE SELECTION

```python
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
df=pd.read_csv("/content/income(1) (1).csv",na_values=[" ?"])
df
```
![alt text](<Screenshot 2024-10-08 111214.png>)

```python
df.isnull().sum()
```
![alt text](<Screenshot 2024-10-08 111220.png>)

```python
missing=df[df.isnull().any(axis=1)]
missing
```
![alt text](<Screenshot 2024-10-08 111234.png>)

```python
df2=df.dropna(axis=0)
df2
```
![alt text](<Screenshot 2024-10-08 111214-1.png>)

```python
sal=df['SalStat']
df2['SalStat']=df2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(df2['SalStat'])
df2
```
![alt text](<Screenshot 2024-10-08 111302.png>)

![alt text](<Screenshot 2024-10-08 111316.png>)

```python
new_data=pd.get_dummies(df2,drop_first=True)
new_data
```
![alt text](<Screenshot 2024-10-08 111412.png>)

```python
columns_list=list(new_data.columns)
print(columns_list)
```
![alt text](<Screenshot 2024-10-08 111441.png>)

```python
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![alt text](<Screenshot 2024-10-08 111455.png>)

```python
x=new_data[features].values
print(x)
```
![alt text](<Screenshot 2024-10-08 111501.png>)

```python
y=new_data['SalStat'].values
print(y)
```
![alt text](<Screenshot 2024-10-08 111505.png>)

```python
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x,train_y)
```
![alt text](<Screenshot 2024-10-08 111510.png>)

```python
prediction=KNN_classifier.predict(test_x)
print(prediction)
```
![alt text](<Screenshot 2024-10-08 111516.png>)

```python
confusionMmatrix=confusion_matrix(test_y,prediction)
print(confusionMmatrix)
```
![alt text](<Screenshot 2024-10-08 111526.png>)

```python
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![alt text](<Screenshot 2024-10-08 111533.png>)

```python
print('Misclassified samples: %d' %(test_y!=prediction).sum())
```
![alt text](<Screenshot 2024-10-08 111538.png>)

```python
df.shape
```
![alt text](<Screenshot 2024-10-08 111545.png>)

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![alt text](<Screenshot 2024-10-08 111550.png>)

```python
contigency_table=pd.crosstab(tips['sex'],tips['time'])
print(contigency_table)
```
![alt text](<Screenshot 2024-10-08 111555.png>)

```python
chi2, p, _, _=chi2_contingency(contigency_table)
print(f"Chi-square statistics:{chi2}")
print(f"p-value:{p}") 
```
![alt text](<Screenshot 2024-10-08 111601.png>)

```python
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif,f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
df
```
![alt text](<Screenshot 2024-10-08 111606.png>)

```python
x=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![alt text](<Screenshot 2024-10-08 111616.png>)
# RESULT:
          Hence,Feature Scaling and Feature Selection process has been performed on the given data set.

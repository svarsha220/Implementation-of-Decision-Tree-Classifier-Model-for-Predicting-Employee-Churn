# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

STEP 1 : Start the program
<br>
STEP 2 : Attach the given data file
<br>
STEP 3 : Now find the satisfaction level of employee data
<br>
STEP 4 : Find the accuracy and new predict value
<br>
STEP 5 : End the program


## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: varsha s
RegisterNumber: 212222220055

import pandas as pd
data=pd.read_csv(r"C:\Users\admin\Downloads\Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours",
"time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
![Screenshot 2024-09-27 132753](https://github.com/user-attachments/assets/fd0f0f22-b845-4f7e-b8d3-0124d0229261)


### Accuracy
![Screenshot 2024-09-27 132826](https://github.com/user-attachments/assets/79a0b1fc-3d85-4168-ac1a-48aba648123f)


### New Predicted  
![image](https://github.com/user-attachments/assets/e132573b-2498-48d7-8ee7-1b1317dd20b4)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

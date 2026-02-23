import pandas as pd

# Load CSV with semicolon separator
data = pd.read_csv("data/student-mat.csv", sep=';')

# Show first 5 rows
print("\nFirst 5 rows:")
print(data.head())

# Show columns and shape
print("\nColumns:", data.columns)
print("\nShape:", data.shape)

# Quick info
print("\nInfo:")
print(data.info())

# Quick stats
print("\nStatistics:")
print(data.describe())

print(data["age"].value_counts().sort_index())
print(data["sex"].value_counts())
print(data["G3"].mean())
print(data["G3"].min())
print(data["G3"].max())

older_students=data[data["age"]>18]
print("Students older than 18:")
print(older_students)
print("How many:" , older_students.shape[0])

girls=data[data["sex"]=="F"]
boys=data[data["sex"]=="M"]
print("Average G3 for Girls:",girls["G3"].mean())
print("Average G3 for Boys",boys["G3"].mean())
print("Number of girls:", girls.shape[0])
print("Number of boys:", boys.shape[0])


data["passed"]=(data["G3"]>=10).astype(int)
print(data[["G3","passed"]].head(10))
print(data["passed"].value_counts())


data["passed"] = (data["G3"] >= 10).astype(int)
girls = data[data["sex"] == "F"]
boys  = data[data["sex"] == "M"]
print("Pass rate:", data["passed"].mean())
print("Girls pass rate:", girls["passed"].mean())
print("Boys pass rate:", boys["passed"].mean())
print(girls["passed"].value_counts())
print(boys["passed"].value_counts())
print(data.isnull().sum())

data["sex_num"]=data["sex"].map({"F":0 ,"M":1})
print(data[["sex","sex_num"]].head(10))

data["school_num"]=data["school"].map({"GP":0 , "MS":1})
print(data[["school","school_num"]].head(10))

data["address_num"]=data["address"].map({"U":0 ,"R":1})
print(data[["address","address_num"]].head(10))

data["famsize_num"]=data["famsize"].map({"GT3":0,"LE3":1})
print(data[["famsize","famsize_num"]].head(10))

data["Pstatus_num"]=data["Pstatus"].map({"T":0,"A":1})
print(data[["Pstatus","Pstatus_num"]].head(10))

data["schoolsup_num"]=data["schoolsup"].map({"yes":1 ,"no":0})
print(data[["schoolsup","schoolsup_num"]].head(10))

data["famsup_num"]=data["famsup"].map({"yes":1,"no":0})
print(data[["famsup","famsup_num"]].head(10))

data["paid_num"]=data["paid"].map({"yes":1 , "no":0})
print(data[["paid","paid_num"]].head(10))

data["activities_num"]=data["activities"].map({"yes":1,"no":0})
print(data[["activities","activities_num"]].head(10))

data["nursery_num"]=data["nursery"].map({"yes":1,"no":0})
print(data[["nursery","nursery_num"]].head(10))

data["higher_num"]=data["higher"].map({"yes":1,"no":0})
print(data[["higher","higher_num"]].head(10))

data["internet_num"]=data["internet"].map({"yes":1,"no":0})
print(data[["internet","internet_num"]].head(10))

data["romantic_num"]=data["romantic"].map({"yes":1,"no":0})
print(data[["romantic","romantic_num"]].head(10))

# تبدیل شغل والدین به عدد
data["Mjob_num"] = data["Mjob"].map({
    "at_home": 0,
    "teacher": 1,
    "health": 2,
    "services": 3,
    "other": 4
})

data["Fjob_num"] = data["Fjob"].map({
    "at_home": 0,
    "teacher": 1,
    "health": 2,
    "services": 3,
    "other": 4
})

# دلیل انتخاب مدرسه
data["reason_num"] = data["reason"].map({
    "home": 0,
    "reputation": 1,
    "course": 2,
    "other": 3
})

# سرپرست دانش‌آموز
data["guardian_num"] = data["guardian"].map({
    "mother": 0,
    "father": 1,
    "other": 2
})


# Features = all numeric columns except 'passed'
feature_cols = [
    "sex_num", "school_num", "address_num", "famsize_num", "Pstatus_num",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures",
    "schoolsup_num", "famsup_num", "paid_num", "activities_num",
    "nursery_num", "higher_num", "internet_num", "romantic_num",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
    "G1", "G2"
]

X = data[feature_cols]
y = data["passed"]
X = data[[
    "sex_num", "school_num", "address_num", "famsize_num", "Pstatus_num",
    "schoolsup_num", "famsup_num", "paid_num", "activities_num", "nursery_num",
    "higher_num", "internet_num", "romantic_num",
    "Mjob_num", "Fjob_num", "reason_num", "guardian_num",
    "traveltime", "studytime", "failures", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences", "G1", "G2"
]]

y = data["passed"]

print(X.head())
print(y.head())


from sklearn.model_selection import train_test_split
X_train, X_test, y_train , y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train = X_train.select_dtypes(include=["int64","float64"])
X_test  = X_test.select_dtypes(include=["int64","float64"])

print("Train size:",X_train.shape[0])
print("Test size:", X_test.shape[0])

from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(
    n_estimators=100 ,
    max_depth=None ,
    random_state=42
)

model.fit(X_train,y_train)
y_pred=model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)
import pandas as pd
importance =model.feature_importances_
feat_imp=pd.Series(importance,index=X.columns)
print(feat_imp.sort_values(ascending=False))
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

Precision=TP / (TP + FP)
Recall=TP / (TP + FN)
F1=2*Precision*Recall/(Precision+Recall)
print("Precision:",Precision)
print("Recall:",Recall)
print("F1_score:",F1)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


import pandas as pd
importance =model.feature_importances_
feature_names=X.columns
feat_imp=pd.Series(importance,index=feature_names)
print(feat_imp.sort_values(ascending=False))

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
scaler= StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

model =LogisticRegression(max_iter=1000)
model.fit(X_train_scaled,y_train)
y_pred=model.predict(X_test_scaled)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X_train, y_train)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(X_train , y_train)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
model_lr=LogisticRegression(max_iter=1000)
model_lr.fit(X_train_scaled,y_train)
y_pred_lr=model_lr.predict(X_test_scaled)

from sklearn.metrics import accuracy_score , confusion_matrix,classification_report
print("Logistic Regression Accuracy:",accuracy_score(y_test,y_pred_lr))
print("Confusion Matrix:/n",confusion_matrix(y_test,y_pred_lr))
print(classification_report(y_test,y_pred_lr))

from sklearn.tree import DecisionTreeClassifier
model_dt=DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train,y_train)
y_pred_dt=model_dt.predict(X_test)

from sklearn.neighbors import KNeighborsClassifier
model_knn=KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train_scaled,y_train)
y_pred_knn=model_knn.predict(X_test_scaled)



from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ===== Decision Tree =====
dt_model = DecisionTreeClassifier(max_depth=5,min_samples_split=10,min_samples_leaf=5,random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
cm_dt = confusion_matrix(y_test, y_pred_dt)

print("Decision Tree Accuracy:", accuracy_dt)
print("Decision Tree F1 Score:", f1_dt)
print("Decision Tree Confusion Matrix:\n", cm_dt)


# ===== KNN =====
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)
cm_knn = confusion_matrix(y_test, y_pred_knn)

print("KNN Accuracy:", accuracy_knn)
print("KNN F1 Score:", f1_knn)
print("KNN Confusion Matrix:\n", cm_knn)




# بعد از fit کردن Random Forest
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

importances = model_rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns)
print(feat_imp.sort_values(ascending=False))

#nemoodar feature importance
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

importances = model_rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns)
feat_imp_sorted = feat_imp.sort_values(ascending=False)


feat_imp_sorted.plot(kind='bar', figsize=(12,6))
plt.title("Random Forest Feature Importance")
plt.ylabel("Importance")
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Random Forest
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# Logistic Regression (نیاز به scaling)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train_scaled, y_train)
y_pred_lr = model_lr.predict(X_test_scaled)

# Decision Tree
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)

# KNN
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train_scaled, y_train)
y_pred_knn = model_knn.predict(X_test_scaled)

results = []



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1,3, figsize=(15,4))

for i, col in enumerate(["G1","G2","G3"]):
    axs[i].hist(data[col], bins=15)
    axs[i].set_title(col)
    axs[i].set_xlabel("Score")
    axs[i].set_ylabel("Count")

plt.tight_layout()
plt.show()

import seaborn as sns

plt.figure(figsize=(7,5))
sns.boxplot(data=data[["G1","G2","G3"]])
plt.title("Score Distribution Comparison")
plt.ylabel("Score")
plt.show()

plt.figure(figsize=(6,5))
plt.scatter(data["G2"], data["G3"], alpha=0.6)
plt.xlabel("G2")
plt.ylabel("G3")
plt.title("G2 vs G3")
plt.show()

sns.countplot(x="passed", data=data)
plt.title("Pass vs Fail Distribution")
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_lr,
    display_labels=["Fail","Pass"]
)
plt.title("Confusion Matrix — Logistic Regression")
plt.show()

import pandas as pd
from sklearn.metrics import accuracy_score

acc_rf   = accuracy_score(y_test, y_pred_rf)
acc_lr   = accuracy_score(y_test, y_pred_lr)
acc_dt   = accuracy_score(y_test, y_pred_dt)
acc_knn  = accuracy_score(y_test, y_pred_knn)
model_names = ["Random Forest","Logistic Regression","Decision Tree","KNN"]
accuracies = [acc_rf, acc_lr, acc_dt, acc_knn]

plt.figure(figsize=(8,5))
plt.bar(model_names, accuracies)
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.xticks(rotation=20)
plt.show()

feat_imp_sorted.plot(kind='bar', figsize=(10,5))
plt.title("Random Forest Feature Importance")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.scatter(data["G2"], data["G3"], alpha=0.6, color="orange")
plt.xlabel("G2")
plt.ylabel("G3")
plt.title("G2 vs G3")
plt.plot([0,20],[0,20], color="red", linestyle="--")

# مهم: ذخیره کردن نمودار
plt.savefig("G2_vs_G3.png")  # فایل PNG داخل پوشه پروژه ذخیره می‌شود
plt.close()  # بستن نمودار تا حافظه آزاد شود

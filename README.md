🎓Student Performance Prediction using Machine Learning

📌 Overview

This project aims to predict students’ final academic performance using Machine Learning techniques.
By analyzing demographic, social, and academic factors, the model identifies patterns that influence student success and failure.

The project compares multiple classification models and evaluates their performance to determine the most effective approach.



## 🎯 Objectives

* Predict whether a student will **pass or fail**
* Identify the most important factors affecting performance
* Compare different machine learning algorithms
* Analyze the impact of prior grades (G1, G2) on final grade (G3)

---

## 📊 Dataset

The dataset contains information about students’:

* Academic history (G1, G2, G3)
* Study habits
* Family background
* Social activities
* Lifestyle factors
* School support variables

### 🔑 Target Variable

**passed**

* 1 → Pass
* 0 → Fail

---

## 🤖 Models Used

The following classification models were implemented and compared:

* 🌲 Random Forest
* 📈 Logistic Regression
* 🌳 Decision Tree
* 📍 K-Nearest Neighbors (KNN)

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing

* Handling categorical variables
* Encoding features
* Feature selection
* Train-test split
* Normalization (when required)

---

### 2️⃣ Feature Importance Analysis

Random Forest was used to determine which features most strongly influence student outcomes.

Key findings:

* Previous grades (G1, G2) are the strongest predictors
* Behavioral and social factors have smaller but noticeable effects

---

### 3️⃣ Model Evaluation Metrics

Each model was evaluated using:

* Accuracy
* F1 Score
* Confusion Matrix

---

## 📈 Results

Logistic Regression achieved the highest accuracy among the tested models, indicating that the dataset is relatively linearly separable.

The results confirm that early academic performance is a strong indicator of final outcomes.

---

## 📉 Visualizations

The project includes several visual analyses:

* Grade distributions (G1, G2, G3)
* Feature importance chart
* Confusion matrix
* Scatter plot of G2 vs G3
* Pass/Fail distribution by gender

---

## 🧠 Key Insights

* Previous grades are the dominant predictors of final performance
* Removing G1 and G2 significantly reduces model accuracy
* The dataset shows moderate class imbalance
* No severe overfitting was observed

---

## 🛠️ Technologies Used

* Python
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn

---

## 🚀 How to Run the Project

1. Clone the repository

```bash
git clone https://github.com/your-username/student-performance-ml.git
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the main script

```bash
python main.py
```







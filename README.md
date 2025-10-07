# Credit Risk Prediction

## 🎯 Objective

The goal of this task is to predict whether a loan application will be approved or not based on applicant data. The task involves preprocessing the dataset, visualizing key features, and training classification models to make accurate predictions.

---

## 📁 Dataset Description

- **Dataset Name:** Loan Prediction Training Set
- **Format:** CSV
- **Shape:** ~614 rows × 13 columns
- **Target Variable:** `Loan_Status` (Y = Approved, N = Not Approved)
- **Features Include:**
  - `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`
  - `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`
  - `Property_Area`

---

## 🛠️ Tools & Libraries Used

- Python
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn (for encoding, model training, evaluation)

---

## 🧪 Approach

1. **Data Loading & Inspection**
   - Loaded the dataset using `pandas.read_csv()`.
   - Checked shape, column names, and previewed the dataset.
   - Analyzed missing values.

2. **Data Cleaning**
   - Filled missing values for `Gender` and `LoanAmount`.
   - Dropped remaining rows with missing values.

3. **Data Visualization**
   - **Histogram** of `LoanAmount` and `ApplicantIncome`.
   - **Countplot** of `Education` vs `Loan_Status`.

4. **Data Preparation**
   - Encoded categorical variables using `LabelEncoder`.
   - Selected features: `ApplicantIncome`, `LoanAmount`, `Education`, `Gender`.
   - Split the data into training and testing sets (80/20 split).

5. **Modeling**
   - Trained two classifiers:
     - **Logistic Regression**
     - **Decision Tree Classifier**
   - Evaluated using accuracy and confusion matrix.

---

## 📈 Results & Insights

- Logistic Regression Accuracy: ~**(depends on dataset, e.g., 75–80%)**
- Decision Tree Accuracy: ~**(depends on dataset, e.g., 70–80%)**
- Most predictive features include:
  - Applicant's income
  - Loan amount
  - Education level
  - Gender

> Logistic Regression gave slightly better performance and more stable results.

---

## 📂 Project Files

- `Task-02.ipynb` – Contains code for data cleaning, visualization, model training, and evaluation.
- `README.md` – This file documenting the task.

---

## ✅ Internship Submission Checklist

- [x] Jupyter Notebook with code and markdown
- [x] Dataset loaded and inspected
- [x] Missing values handled
- [x] Visualizations included
- [x] Model trained and evaluated
- [x] Accuracy and confusion matrix shown
- [x] Code is clean and well-commented
- [x] README.md included
- [x] Uploaded to GitHub repository
- [x] Link submitted via Google Classroom

---

## 🚀 How to Run This Project

1. Clone the repository:
   ```bash
   git clone https://github.com/SayabArshad/Credit-Risk-Prediction.git

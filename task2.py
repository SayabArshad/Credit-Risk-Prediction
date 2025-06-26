# Task 2 : Credit Risk Prediction
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# path of dataset
df = pd.read_csv("D:/python_ka_chilla/Internship/task 2/train.csv")  # <-- Make sure the file path is correct

print(df.shape)
print(df.columns)
print(df.head())
# Show missing values
print(df.isnull().sum())

# Fill or drop missing data
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df.dropna(inplace=True)  # or apply smart filling for all

print(df.isnull().sum())  # Recheck


# Loan Amount distribution
sns.histplot(df['LoanAmount'], kde=True)
plt.title("Loan Amount Distribution")
plt.show()

# Education vs Loan Status
sns.countplot(data=df, x='Education', hue='Loan_Status')
plt.title("Education vs Loan Status")
plt.show()

# Applicant Income distribution
sns.histplot(df['ApplicantIncome'], kde=True)
plt.title("Applicant Income Distribution")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
df_encoded = df.copy()
le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Features & labels
X = df_encoded[['ApplicantIncome', 'LoanAmount', 'Education', 'Gender']]
y = df_encoded['Loan_Status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))

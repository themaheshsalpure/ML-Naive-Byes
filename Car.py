# Import necessary libraries
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Load the dataset
dataset = pd.read_csv("C:/Data Science/datasets/NB_Car_Ad.csv")

# Data Cleaning
# No explicit mention of missing values, so assuming the dataset is clean
# Remove unnecessary columns (User ID)
dataset.drop("User ID", axis=1, inplace=True)

# Feature Engineering
# No explicit feature engineering mentioned in the problem statement

# Exploratory Data Analysis (EDA)
# Summary
print(dataset.describe())

# Univariate Analysis
# Distribution plots
sns.distplot(dataset['Age'])
sns.distplot(dataset['EstimatedSalary'])

# Count plot for Gender
sns.countplot(x='Gender', data=dataset)

# Bar plot for Purchased
sns.countplot(x='Purchased', data=dataset)

# Bivariate Analysis
# Scatter plots
sns.scatterplot(x='Age', y='EstimatedSalary', hue='Purchased', data=dataset)

# Box plot or violin plot
sns.boxplot(x='Gender', y='EstimatedSalary', data=dataset)

# Model Building
# Separate features and target variable
X = dataset.drop('Purchased', axis=1)
y = dataset['Purchased']

# Encode categorical feature (Gender)
X = pd.get_dummies(X, columns=['Gender'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a Naïve Bayes Model
naive_bayes_model = BernoulliNB()
naive_bayes_model.fit(X_train_scaled, y_train)

# Validate the Model
# Predict on the test set
y_pred = naive_bayes_model.predict(X_test_scaled)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Print evaluation metrics
print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Print the accuracy in percentage format
print(f'\nAccuracy: {accuracy * 100:.2f}%')  # Print accuracy

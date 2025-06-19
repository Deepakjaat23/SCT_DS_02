import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the built-in Titanic dataset
df = sns.load_dataset('titanic')

# Show basic info
print("Basic Info:")
print(df.info(), "\n")

# Show first few rows
print("Head of Dataset:")
print(df.head(), "\n")

# Descriptive statistics
print("Descriptive Statistics:")
print(df.describe(include='all'), "\n")

# Check for duplicates
print("Total Duplicate Rows:", df.duplicated().sum(), "\n")

# Unique values in categorical columns
print("Unique values:")
print("Pclass:", df['pclass'].unique())
print("Survived:", df['survived'].unique())
print("Sex:", df['sex'].unique(), "\n")

# Null values before cleaning
print("Missing Values Before Cleaning:")
print(df.isnull().sum(), "\n")

# Fill 'deck' NaNs with 'Unknown' and convert to string
df['deck'] = df['deck'].astype(str).replace('nan', 'Unknown')

# Optionally fill other columns – better to use median/mode instead of '0'
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df['embark_town'].fillna(df['embark_town'].mode()[0], inplace=True)
df['alive'].fillna('no', inplace=True)
df['who'].fillna('unknown', inplace=True)
df['adult_male'].fillna(False, inplace=True)

# Confirm all nulls handled
print("Missing Values After Cleaning:")
print(df.isnull().sum(), "\n")


# Passenger class count
sns.countplot(x='pclass', data=df)
plt.title("Passenger Class Count")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# Survival distribution pie chart
df['survived'].value_counts().plot.pie(
    autopct='%1.1f%%', colors=['red', 'green'], labels=['Not Survived', 'Survived'])
plt.title('Survival Distribution')
plt.ylabel('')
plt.show()

# Age distribution histogram
sns.histplot(df['age'], bins=20, kde=True, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Survival rate by gender
sns.barplot(x='sex', y='survived', data=df, palette='pastel')
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.show()

# Survival rate by class
sns.barplot(x='pclass', y='survived', data=df, palette='Set2')
plt.title('Survival Rate by Class')
plt.xlabel('Class')
plt.ylabel('Survival Rate')
plt.show()

# Violin plot: age vs survival
sns.violinplot(x='survived', y='age', data=df, palette='muted')
plt.title('Age Distribution by Survival')
plt.xlabel('Survived')
plt.ylabel('Age')
plt.show()

# Correlation heatmap
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Grouped bar: survival by class and gender
sns.barplot(x='pclass', y='survived', hue='sex', data=df, palette='Set1')
plt.title('Survival Rate by Class and Gender')
plt.xlabel('Class')
plt.ylabel('Survival Rate')
plt.legend(title='Gender')
plt.show()

# Scatter plot: age vs fare colored by class and styled
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='age', y='fare', hue='class', style='sex', palette='deep')

plt.title('Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.grid(True)
plt.legend(title='Class / Sex')
plt.tight_layout()
plt.show()
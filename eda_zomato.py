# --------------------------------------------------
# 📌 IMPORT LIBRARIES
# --------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')


# --------------------------------------------------
# 📌 LOAD DATA
# --------------------------------------------------
df = pd.read_csv("zomato.csv", encoding='latin1')

print("\n--- DATA LOADED SUCCESSFULLY ---")


# --------------------------------------------------
# 📌 BASIC INFORMATION
# --------------------------------------------------
print("\n--- BASIC INFO ---")
print(df.info())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- NUMERIC SUMMARY ---")
print(df.describe())

print("\n--- CATEGORICAL SUMMARY ---")
print(df.describe(include='object'))


# --------------------------------------------------
# 📌 CLEANING DATA
# --------------------------------------------------

# Remove duplicates
df.drop_duplicates(inplace=True)

# Replace missing cuisines
df['Cuisines'] = df['Cuisines'].fillna("Unknown")


# --------------------------------------------------
# 📌 RATING DISTRIBUTION
# --------------------------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df['Aggregate rating'], bins=20, kde=True)
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()


# --------------------------------------------------
# 📌 PRICE RANGE DISTRIBUTION
# --------------------------------------------------
plt.figure(figsize=(7,4))
sns.countplot(x='Price range', data=df)
plt.title("Price Range Distribution")
plt.show()


# --------------------------------------------------
# 📌 VOTES DISTRIBUTION
# --------------------------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df['Votes'], bins=40, kde=True)
plt.title("Votes Distribution")
plt.show()


# --------------------------------------------------
# 📌 ONLINE DELIVERY vs RATING
# --------------------------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(x='Has Online delivery', y='Aggregate rating', data=df)
plt.title("Online Delivery vs Rating")
plt.show()


# --------------------------------------------------
# 📌 TOP CITIES (MOST RESTAURANTS)
# --------------------------------------------------
plt.figure(figsize=(12,6))
df['City'].value_counts().head(15).plot(kind='bar')
plt.title("Top 15 Cities With Most Restaurants")
plt.ylabel("Count")
plt.show()


# --------------------------------------------------
# 📌 TOP CUISINES
# --------------------------------------------------
cuisine_counts = df['Cuisines'].value_counts().head(20)

plt.figure(figsize=(12,6))
sns.barplot(x=cuisine_counts.values, y=cuisine_counts.index)
plt.title("Top 20 Most Common Cuisines")
plt.xlabel("Count")
plt.ylabel("Cuisine")
plt.show()


# --------------------------------------------------
# 📌 COST vs RATING
# --------------------------------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(x='Average Cost for two', y='Aggregate rating', data=df)
plt.title("Cost for Two vs Ratings")
plt.show()


# --------------------------------------------------
# 📌 CORRELATION HEATMAP (FIXED)
# --------------------------------------------------
plt.figure(figsize=(10,7))

numeric_df = df.select_dtypes(include=['int64', 'float64'])

sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# --------------------------------------------------
# 📌 INSIGHTS (TEXT OUTPUT)
# --------------------------------------------------
print("\n--- TOP INSIGHTS ---")

print("\nTop 10 Most Common Cuisines:")
print(df['Cuisines'].value_counts().head(10))

print("\nTop 10 Cities With Most Restaurants:")
print(df['City'].value_counts().head(10))

print("\nAverage Rating by Price Range:")
print(df.groupby('Price range')['Aggregate rating'].mean())

print("\nAverage Cost for Two by City:")
print(df.groupby('City')['Average Cost for two'].mean().sort_values(ascending=False).head(10))

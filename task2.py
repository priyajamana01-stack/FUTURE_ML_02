import pandas as pd
import re

# STEP 1: Load the dataset
df = pd.read_csv("customer_support_tickets.csv")

# STEP 2: Clean column names
df.columns = df.columns.str.strip()

# STEP 3: Create text column by combining subject + description
df['text'] = df['Ticket Subject'].astype(str) + " " + df['Ticket Description'].astype(str)

# STEP 4: Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\{.*?\}', '', text)      # remove {placeholders}
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply text cleaning
df['clean_text'] = df['text'].apply(clean_text)

# Final output check
print(df[['clean_text', 'Ticket Type']].head())

import matplotlib.pyplot as plt

# Count of tickets by type
ticket_counts = df['Ticket Type'].value_counts()

print(ticket_counts)

# Plot
plt.figure(figsize=(8,5))
ticket_counts.plot(kind='bar', color='skyblue')
plt.title("Ticket Type Distribution")
plt.xlabel("Ticket Type")
plt.ylabel("Number of Tickets")
plt.grid(axis='y')
plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

X = vectorizer.fit_transform(df['clean_text'])

# Target column (what we want to predict)
y = df['Ticket Type']

print("Text converted to numbers successfully")
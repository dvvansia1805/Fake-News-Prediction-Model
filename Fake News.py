import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

# Print the list of stopwords
print(stopwords.words('english'))

# Load the dataset using a properly formatted path
news_dataset = pd.read_csv(r'D:\College Work\AI\Fake News Prediction\train.csv')

# Print the shape of the dataset
print("Shape of the dataset:", news_dataset.shape)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(news_dataset.head())

# Check for missing values
print("The number of missing values in the dataset:")
print(news_dataset.isnull().sum())

# Fill missing values in 'author' and 'title' columns with empty strings
news_dataset['author'].fillna('', inplace=True)
news_dataset['title'].fillna('', inplace=True)

# Combine the author and title columns to create a single text column
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

# Display the first few rows of the new dataset with the content column
print("First few rows of the dataset with the new content column:")
print(news_dataset[['content']].head())

# Data preprocessing
# Initialize the PorterStemmer
port_stem = PorterStemmer()

# Function to preprocess the text
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  # Removing all the special characters and numbers
    text = text.lower()  # Converting to lowercase
    text = text.split()  # Splitting into words
    text = [port_stem.stem(word) for word in text if not word in stopwords.words('english')]  # Stemming and removing stopwords
    text = ' '.join(text)  # Joining the words back to form a single string
    return text

# Apply the preprocess_text function to the content column
news_dataset['content'] = news_dataset['content'].apply(preprocess_text)

# Display the first few rows of the dataset after preprocessing
print("First few rows of the dataset after preprocessing:")
print(news_dataset[['content']].head())

# Separating the data and label
X = news_dataset['content'].values
y = news_dataset['label'].values  # Assuming there's a 'label' column

# Convert the textual data to numerical data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy score of test data
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

X_new = X_test[0]
def predict_news(model, X_new):
    # Make predictions
    prediction = model.predict(X_new)
    
    # Print the prediction
    print("Prediction:", prediction[0])
    
    # Interpret the prediction
    if prediction[0] == 0:
        print('The news is Real')
    else:
        print('The news is Fake')

# Example usage:
# Assuming X_new is a single new data point
X_new = vectorizer.transform(["New news article to predict"])
predict_news(model, X_new)


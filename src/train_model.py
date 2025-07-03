# src/train_model.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Step 1: Load the dataset
df = pd.read_csv(r'D:\projects\emoji_predictor\.venv\data\emoji_data.csv')

# Step 2: Split into features (X) and labels (y)
X = df['text']  # Sentences
y = df['emoji']  # Corresponding emojis

# Step 3: Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create a pipeline - vectorize text + classify
model = Pipeline([
    ('vectorizer', CountVectorizer()),  # Converts text to numbers
    ('classifier', MultinomialNB())     # Naive Bayes Classifier
])

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Test the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 7: Save the model
joblib.dump(model, 'emoji_model.pkl')
print("✅ Model saved as emoji_model.pkl")

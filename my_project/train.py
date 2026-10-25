import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# 1. Create a simple toy dataset for sentiment analysis
data = {
    'text': [
        'I love this product, it is amazing!',
        'Absolute worst experience ever, hate it.',
        'Pretty good, quite happy with the quality.',
        'Terrible customer service, total waste of money.',
        'Amazing and wonderful, exceeded my expectations!'
    ],
    'label': ['positive', 'negative', 'positive', 'negative', 'positive']
}

df = pd.DataFrame(data)

# 2. Convert text data into numerical features using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# 3. Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X, y)

# 4. Save both the trained model and vectorizer using joblib
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model trained and saved successfully!")
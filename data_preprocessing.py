import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Data Preprocessing
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test.dropna(subset=['text'], inplace=True)

# Drop any rows with missing text or label
train = train.dropna(subset=['text', 'label'])

# 2. Model Training
# Tokenize and Vectorize
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(train['text'])
tfidf_test = tfidf_vectorizer.transform(test['text'])

# Splitting the data
X_train, X_val, y_train, y_val = train_test_split(tfidf_train, train['label'], test_size=0.2, random_state=42)

# Training a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)

# Printing the classification report
print(classification_report(y_val, y_pred))

print(f"Validation Accuracy: {accuracy_score(y_val, y_pred)}")

y_test_pred = clf.predict(tfidf_test)


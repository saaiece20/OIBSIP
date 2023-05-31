import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
email_data = pd.read_csv('path_to_dataset/spam_email_data.csv')
features = email_data['content']
labels = email_data['label']
tfidf_vectorizer = TfidfVectorizer()
features_tfidf = tfidf_vectorizer.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(features_tfidf, labels, test_size=0.2, random_state=42)
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

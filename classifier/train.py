from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are downloaded


# Fetch the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', categories=['rec.sport.baseball', 'sci.space', 'talk.politics.mideast', 'comp.graphics', 'sci.med'])
# texts = newsgroups.data[:1000]  # Use only the first 1000 articles
# labels = newsgroups.target[:1000]

texts = newsgroups.data
labels = newsgroups.target

# Preprocess the dataset
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

preprocessed_texts = [preprocess_text(text) for text in texts]

# Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(preprocessed_texts)
y = labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
 
# Save the model and vectorizer
joblib.dump(model, 'pkl/news_category_model.pkl')
joblib.dump(vectorizer, 'pkl/news_category_vectorizer.pkl')
joblib.dump(newsgroups.target_names, 'pkl/target_names.pkl')







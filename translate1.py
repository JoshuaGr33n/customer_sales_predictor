import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import pickle

# Load the train, test, and validation datasets
train_data = pd.read_csv('CSV/translate/train.csv')
test_data = pd.read_csv('CSV/translate/test.csv')
validate_data = pd.read_csv('CSV/translate/validation.csv')

# Combine the datasets (if needed)
data = pd.concat([train_data, validate_data])

# Features and target variable
X = data['en']  # English sentences
y = data['zh']  # Chinese sentences

# Split the data (using the test set for evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Further limit the number of features
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model using SGDClassifier
model = SGDClassifier()
model.fit(X_train_vec, y_train)

# Save the model and vectorizer to .pkl files
with open('models/translate/translation_model.pkl', 'wb') as file:
    pickle.dump(model, file)
with open('models/translate/vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)
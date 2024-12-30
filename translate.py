import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import pickle

# Sample data
data = {
    'en': ["Hello", "How are you?", "Good morning", "Good night", "Thank you", "Yes", "No", "Please", "Sorry", "Excuse me"],
    'zh': ["你好", "你好吗？", "早上好", "晚安", "谢谢", "是", "不", "请", "对不起", "打扰一下"]
}



# Convert to DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df['en']  # English sentences
y = df['zh']  # Chinese sentences

# Split the data (using the test set for evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
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

# Function to translate new English sentences
def translate_sentences(sentences):
    # Vectorize the input sentences
    sentences_vec = vectorizer.transform(sentences)
    # Predict the translations
    translations = model.predict(sentences_vec)
    return translations

# Example usage
new_sentences = ["Hello", "Good night"]
translations = translate_sentences(new_sentences)

for en, zh in zip(new_sentences, translations):
    print(f"English: {en}")
    print(f"Chinese: {zh}")
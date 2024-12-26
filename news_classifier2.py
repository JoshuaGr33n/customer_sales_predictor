import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure NLTK resources are downloaded
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def fetch_articles(url, article_tag, title_tag, content_tag, title_class=None, content_class=None):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = []
        for item in soup.find_all(article_tag):
            title_element = item.find(title_tag, class_=title_class) if title_class else item.find(title_tag)
            content_element = item.find(content_tag, class_=content_class) if content_class else item.find(content_tag)
            if title_element and content_element:
                title = title_element.get_text()
                content = content_element.get_text()
                if content:  # Ensure content is not empty
                    articles.append({'title': title, 'content': content})
        return articles
    except requests.exceptions.RequestException as e:
        print(f"Error fetching articles from {url}: {e}")
        return []

# Fetch articles from CNN
cnn_articles = fetch_articles('https://edition.cnn.com/world', 'div', 'span', 'div', title_class='container__headline-text')

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Preprocess CNN articles
preprocessed_articles = [preprocess_text(article['content']) for article in cnn_articles]

# Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(preprocessed_articles)
y = ['unknown'] * len(preprocessed_articles)  # Assuming new articles are unlabeled

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
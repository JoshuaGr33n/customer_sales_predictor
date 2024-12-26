import requests
from bs4 import BeautifulSoup
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are downloaded


# Load the saved model and vectorizer
model = joblib.load('pkl/news_category_model.pkl')
vectorizer = joblib.load('pkl/news_category_vectorizer.pkl')
target_names = joblib.load('pkl/target_names.pkl')


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

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Fetch articles from CNN
cnn_articles = fetch_articles('https://edition.cnn.com/world', 'div', 'span', 'div', title_class='container__headline-text')

# Preprocess the articles
preprocessed_articles = [preprocess_text(article['content']) for article in cnn_articles]

# Vectorize the text data
X = vectorizer.transform(preprocessed_articles)

# Predict the categories
predictions = model.predict(X)

# Map the predicted labels to category names
category_names = target_names
labeled_articles = [{'title': article['title'], 'content': article['content'], 'category': category_names[pred]} for article, pred in zip(cnn_articles, predictions)]

# Print labeled articles
for article in labeled_articles:
    print(f"Title: {article['title']}\nCategory: {article['category']}\n")
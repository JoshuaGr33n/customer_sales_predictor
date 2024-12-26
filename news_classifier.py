import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Ensure NLTK resources are downloaded
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def fetch_full_article(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])
        return content
    except requests.exceptions.RequestException as e:
        print(f"Error fetching article: {e}")
        return ""

def fetch_cnn_articles():
    url = 'https://edition.cnn.com/world'
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = []
        for item in soup.find_all('div', class_='card'):
            title_element = item.find('span', class_='container__headline-text')
            link_element = item.find('a', href=True)
            if title_element and link_element:
                title = title_element.get_text()
                article_url = 'https://edition.cnn.com' + link_element['href']
                content = fetch_full_article(article_url)
                if content:  # Only add articles with content
                    articles.append({'title': title, 'content': content})
        return articles
    except requests.exceptions.RequestException as e:
        print(f"Error fetching CNN articles: {e}")
        return []

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Fetch and preprocess CNN articles
cnn_articles = fetch_cnn_articles()
preprocessed_articles = [preprocess_text(article['content']) for article in cnn_articles]

# Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(preprocessed_articles)

# Dummy labels for demonstration purposes
y = ['class1' if i % 2 == 0 else 'class2' for i in range(len(preprocessed_articles))]

# Perform cross-validation with Logistic Regression
model = LogisticRegression(max_iter=1000)
scores = cross_val_score(model, X, y, cv=5)
print(f'Cross-Validation Accuracy: {scores.mean()}')
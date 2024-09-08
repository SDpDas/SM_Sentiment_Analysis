import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the saved model and vectorizer
loaded_model = pickle.load(open('trained_model.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.sav', 'rb'))

# Define the preprocessing function used during training
port_stem = PorterStemmer()
def preprocess_text(text):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', text)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Example tweets to predict
tweets = ["I am never going to buy this product again", "I love using this new feature!", "Awful service it was and the staff had rude behaviour"]

# Preprocess the tweets
preprocessed_tweets = [preprocess_text(tweet) for tweet in tweets]

# Transform the tweets using the loaded vectorizer
X_new = vectorizer.transform(preprocessed_tweets)

# Make predictions
predictions = loaded_model.predict(X_new)

# Print predictions
for tweet, prediction in zip(tweets, predictions):
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    print(f"Tweet: {tweet}\nPrediction: {sentiment}\n")

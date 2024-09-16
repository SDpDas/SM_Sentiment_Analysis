import pickle 
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#Loading the model and vectorizer
load_model = pickle.load(open('/trained_model/trained_modelv2.sav', 'rb'))
vectorizer = pickle.load(open('/vectorizer/vectorizerv2.sav', 'rb'))

#Preprocessing the data
port_stem = PorterStemmer()
def preprocess_text(text):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', text)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

#Example tweets
tweets = ["This was an exciting journey", "Shame she had to leave us", "I don't like you."]

#preprocess the tweet
preprocess_tweets = [preprocess_text(tweet) for tweet in tweets]

#Transform the chats using vectorizer
X_new = vectorizer.transform(preprocess_tweets)

#Make predictions
predictions = load_model.predict(X_new)

#0 for negative, 1 for neutral, 2 for positive chat

#Print predictions
for chat, prediction in zip(tweets, predictions):
    if prediction == 4:
        Sentiment = 'Positive Chat'
    elif prediction == 2:
        Sentiment = 'Neutral Chat'
    elif prediction == 0:
        Sentiment = 'Negative Chat'

    print(f"Pred_value: {prediction}\nChat: {chat}\nPrediction: {Sentiment}\n")
import pickle 
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#Loading the model and vectorizer
load_model = pickle.load(open('/trained_model/dreview_model.sav', 'rb'))
vectorizer = pickle.load(open('/vectorizer/dvectorizer.sav', 'rb'))

#Preprocessing the data
port_stem = PorterStemmer()
def preprocess_text(text):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', text)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

#eg. reviews
reviews = ["Very poor medicine and my condition worsened.", "I'm never coming back to this hospital again", "It cured me in 3 days.", "Thanks to the whole doctor team."]

#preprocess the reviews
preprocess_reviews = [preprocess_text(review) for review in reviews]

#Transform the reviews using vectorizer
X_new = vectorizer.transform(preprocess_reviews)

#Make predictions
predictions = load_model.predict(X_new)

#0 for negative, 1 for neutral, 2 for positive chat

#Print predictions
for chat, prediction in zip(reviews, predictions):
    if prediction == 2:
        Sentiment = 'Positive Rating'
    elif prediction == 1:
        Sentiment = 'Neutral Rating'
    elif prediction == 0:
        Sentiment = 'Negative Rating'

    print(f"Pred_value: {prediction}\nReview: {chat}\nPrediction: {Sentiment}\n")
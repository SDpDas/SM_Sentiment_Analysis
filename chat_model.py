import pickle 
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#Loading the model and vectorizer
load_model = pickle.load(open('/trained_model/chat_trained_model.sav', 'rb'))
vectorizer = pickle.load(open('/vectorizer/chat_vectorizer.sav', 'rb'))

#Preprocessing the data
port_stem = PorterStemmer()
def preprocess_text(text):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', text)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

#Example chats
chats = ["I hate you.", "You are the best.", "I won't be coming home today."]

#preprocess the chats
preprocess_chats = [preprocess_text(chat) for chat in chats]

#Transform the chats using vectorizer
X_new = vectorizer.transform(preprocess_chats)

#Make predictions
predictions = load_model.predict(X_new)

#0 for negative, 1 for neutral, 2 for positive chat

#Print predictions
for chat, prediction in zip(chats, predictions):
    if prediction == 2:
        Sentiment = 'Positive Chat'
    elif prediction == 1:
        Sentiment = 'Neutral Chat'
    elif prediction == 0:
        Sentiment = 'Negative Chat'

    print(f"Pred_value: {prediction}\nChat: {chat}\nPrediction: {Sentiment}\n")
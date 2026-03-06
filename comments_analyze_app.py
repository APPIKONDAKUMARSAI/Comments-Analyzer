import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# load files
model = pickle.load(open("trained_nlp_model.pkl","rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl","rb"))

nltk.download('stopwords')

port_stem = PorterStemmer()

def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [port_stem.stem(word) for word in content if word not in stopwords.words('english')]
    content = ' '.join(content)
    return content


def predict_sentiment(text):

    processed = stemming(text)

    vector = vectorizer.transform([processed])

    prediction = model.predict(vector)

    if prediction[0] == 1:
        return "Positive Tweet 😊"
    else:
        return "Negative Tweet 😠"


# Streamlit UI
st.title("Twitter Sentiment Analysis")

st.write("Enter a tweet to analyze sentiment")

tweet = st.text_area("Tweet Text")

if st.button("Analyze Sentiment"):

    if tweet.strip() != "":
        result = predict_sentiment(tweet)
        st.success(result)
    else:
        st.warning("Please enter a tweet")
import json
import random
import string
import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>
/* Full page pastel background image */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1528459801416-a9e53bbf4e17");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Glassmorphism container */
.chat-container {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 20px;
    max-width: 720px;
    margin: auto;
}

/* User bubble */
.user-bubble {
    background: #c7d2fe;
    color: #1e293b;
    padding: 12px 16px;
    border-radius: 18px;
    margin: 8px 0;
    text-align: right;
}

/* Bot bubble */
.bot-bubble {
    background: #fde2e4;
    color: #1e293b;
    padding: 12px 16px;
    border-radius: 18px;
    margin: 8px 0;
    text-align: left;
}

/* Title */
.title {
    text-align: center;
    font-size: 34px;
    font-weight: bold;
    color: #1e293b;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #475569;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

st.markdown("<div class='title'>🤖 AI Chatbot</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Pastel-themed ML Chat Assistant</div>", unsafe_allow_html=True)


with st.sidebar:
    st.title("🌸 Project Info")
    st.write("**AI / ML Internship Project**")
    st.markdown("""
    **Tech Stack**
    - Python  
    - NLP (NLTK)  
    - TF-IDF  
    - Logistic Regression  
    - Streamlit  
    """)
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

with open("intents.json") as f:
    data = json.load(f)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [w for w in tokens if w not in string.punctuation]
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


sentences = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(clean_text(pattern))
        labels.append(intent["tag"])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

model = LogisticRegression(max_iter=1000)
model.fit(X, labels)


if "messages" not in st.session_state:
    st.session_state.messages = []


st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>🧑 {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    cleaned = clean_text(user_input)
    vect = vectorizer.transform([cleaned])
    intent = model.predict(vect)[0]

    response = "Sorry, I didn't understand that. Please rephrase."

    for i in data["intents"]:
        if i["tag"] == intent:
            response = random.choice(i["responses"])

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    st.rerun()

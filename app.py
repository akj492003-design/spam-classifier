from flask import Flask, request, render_template
import os
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

NLTK_DATA = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.insert(0, NLTK_DATA)

app = Flask(__name__)

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    message = request.form.get("message")
    transformed_message = transform_text(message)
    vector_input = tfidf.transform([transformed_message]).toarray()
    prediction = model.predict(vector_input)[0]
    result = "Spam" if prediction == 1 else "Not Spam"

    return render_template(
        "index.html",
        prediction=result,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)
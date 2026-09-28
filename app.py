from flask import Flask, request, render_template
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


app = Flask(__name__)


# Load the trained model and TF-IDF vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


# Porter Stemmer
ps = PorterStemmer()


def transform_text(text):
    # Lowercase
    text = text.lower()

    # Tokenization
    text = nltk.word_tokenize(text)

    # Keep only alphanumeric words
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Stemming
    for i in text:
        y.append(ps.stem(i))

    # Convert list back to string
    return " ".join(y)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    message = request.form.get('message')

    transformed_message = transform_text(message)

    # Convert text into TF-IDF vector
    vector_input = tfidf.transform([transformed_message])

    # Predict
    prediction = model.predict(vector_input)[0]

    # Convert prediction to readable result
    result = "Spam" if prediction == 1 else "Not Spam"

    return render_template(
        'index.html',
        prediction=result,
        message=message
    )


if __name__ == '__main__':
    app.run(debug=True)
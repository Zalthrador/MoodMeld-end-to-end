import re, numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from flask import (
    Flask,
    render_template,
    send_from_directory,
    redirect,
    url_for,
    request,
    session,
    Response,
)

app = Flask(__name__)
app.secret_key = "your_secret_key"


def tokenizer(text):
    # Convert to lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove special characters
    text = re.sub(r"[^\w\s]", "", text)
    # Remove extra spaces
    text = re.sub(" +", " ", text)
    # Trim leading and trailing whitespaces
    text = text.strip()
    # Tokenize the text into words
    tokens = word_tokenize(text)
    # Initialize the Porter Stemmer
    stemmer = PorterStemmer()
    # Remove stopwords and stem the tokens
    tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in stopwords.words("english")
    ]
    return tokens


import __main__

# Ensure tokenizer is defined or imported above this line
__main__.tokenizer = tokenizer


# Load the saved model
# loaded_model = load("./model.joblib")
# vectorizer = load('./tfidf_vectorizer.joblib')
loaded_pipeline = load("./model_pipeline.joblib")

"""
def compute(text):
    # Builds a TF-IDF matrix for the sentences
    array_of_strings = numpy.array([text], dtype=str)
    # app.logger.info(array_of_strings)
    # app.logger.info(array_of_strings.shape)
    
    tfidf_matrix = vectorizer.transform(array_of_strings)
    # app.logger.info(tfidf_matrix.shape)
    predicted = loaded_model.predict(tfidf_matrix)
    if predicted == 0:
        return "Negative"
    elif predicted == 1:
        return "Neutral"
    elif predicted == 2:
        return "Positive"
"""


def compute(text):
    array_of_strings = numpy.array([text], dtype=str)
    predicted = loaded_pipeline.predict(array_of_strings)
    if predicted == 0:
        return "Negative"
    elif predicted == 1:
        return "Neutral"
    elif predicted == 2:
        return "Positive"



# ROUTIING... & BACKEND...
def check_empty_or_text(input_text):
    if input_text.strip():
        return True
    else:
        return False


@app.route("/", methods=["GET", "POST"])
def base():
    if request.method == "POST":
        msg = request.form.get("message")
        if check_empty_or_text(msg):
            # do the computation and get the result
            mood = compute(msg)
            session["param1"] = mood
            return redirect("/result")
            # return redirect(url_for('get_result'))    <--- same as previous one
            # return render_template("result.html", mmood=mood)
        else:
            return redirect("/")

    elif request.method == "GET":
        # print(request.headers)
        return render_template("base.html")


@app.route("/result", methods=["GET", "POST"])
def get_result():
    if request.method == "GET" and session.get("param1"):
        mood = session["param1"]
        session.clear()  # clear the session for the next use
        return render_template("result.html", mmood=mood)
    elif request.method == "GET":
        return redirect("/")
    elif request.method == "POST":
        return redirect("/")


if __name__ == "__main__":
    app.run(debug=True, port=5000)

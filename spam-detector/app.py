from flask import Flask, render_template, request
import joblib
import numpy as np
import re

app = Flask(__name__)

# Load trained model
model = joblib.load("D:\Internships\Codsoft ML Internship\Spam SMS Detection\spam-detector\spam_classifier_model.pkl")

# Text preprocessing
def preprocess(text):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+",
        " url ",
        text
    )

    text = re.sub(
        r"\b\d+\b",
        " num ",
        text
    )

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text

@app.route("/", methods=["GET", "POST"])

def home():

    prediction = None
    confidence = None
    message = ""

    if request.method == "POST":

        message = request.form["message"]

        clean_msg = preprocess(message)

        pred = model.predict([clean_msg])[0]

        clf = model.named_steps["clf"]

        if hasattr(clf, "predict_proba"):

            confidence = model.predict_proba(
                [clean_msg]
            )[0].max()

        else:

            score = model.decision_function(
                [clean_msg]
            )[0]

            confidence = 1 / (
                1 + np.exp(-abs(score))
            )

        prediction = "Spam" if pred == 1 else "Ham"

        confidence = round(confidence * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        message=message
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
from flask import Flask, render_template, request
import joblib
import numpy as np
import re

app = Flask(__name__)

# Load trained model
model = joblib.load("spam_classifier_model.pkl")

# Text preprocessing
def preprocess(text):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+",
        " url ",
        text
    )

    text = re.sub(
        r"\b\d+\b",
        " num ",
        text
    )

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text

@app.route("/", methods=["GET", "POST"])

def home():

    prediction = None
    confidence = None
    message = ""

    if request.method == "POST":

        message = request.form["message"]

        clean_msg = preprocess(message)

        pred = model.predict([clean_msg])[0]

        clf = model.named_steps["clf"]

        if hasattr(clf, "predict_proba"):

            confidence = model.predict_proba(
                [clean_msg]
            )[0].max()

        else:

            score = model.decision_function(
                [clean_msg]
            )[0]

            confidence = 1 / (
                1 + np.exp(-abs(score))
            )

        prediction = "Spam" if pred == 1 else "Ham"

        confidence = round(confidence * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        message=message
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
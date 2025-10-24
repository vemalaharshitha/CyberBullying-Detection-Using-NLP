from flask import Flask, render_template, request, jsonify
import pickle
import re
import os

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)         # remove links
    text = re.sub(r"@\w+", "", text)            # remove mentions
    text = re.sub(r"[^a-z\s]", "", text)        # keep only letters
    text = re.sub(r"\s+", " ", text).strip()    # remove extra spaces
    return text

# ---------------- ROUTES ---------------- #

# Home → Landing page
@app.route("/")
def home():
    return render_template("landing.html")

# Analyzer page
@app.route("/analyzer")
def analyzer():
    return render_template("analyzer.html")

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]

    # Also get probability score
    prob = model.predict_proba(vec)[0].max() * 100

    result = "Bullying" if pred == 1 else "Not Bullying"
    return jsonify({"prediction": result, "confidence": f"{prob:.2f}%"})

# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    # Debug: confirm Flask sees templates
    print("Template folder:", os.path.abspath(app.template_folder))
    print("Files in templates:", os.listdir(app.template_folder))
    app.run(debug=True)

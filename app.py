import json
from flask import Flask, render_template, request, jsonify
import joblib
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow import keras

# Initialize Flask app
app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load ML pipeline and chatbot components
pipeline = joblib.load('model_pipeline.joblib')  # For cyberbullying classification
model = keras.models.load_model("qna_model.h5")  # Chatbot model

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

with open("label_encoder.pkl", "rb") as le_file:
    label_encoder = pickle.load(le_file)

# Load predefined chatbot questions and answers
def load_chat_data():
    try:
        with open("chat.json", "r") as file:
            data = json.load(file)
        return {item["question"].lower(): item["answer"] for item in data["dataset"]}
    except FileNotFoundError:
        return {}

chat_data = load_chat_data()

# Text preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]
    return " ".join(tokens)

# Chatbot response logic
def get_answer(question):
    processed_question = preprocess_text(question)
    
    # Direct match from chat.json
    if processed_question in chat_data:
        return chat_data[processed_question]
    
    # ML prediction
    vectorized_question = vectorizer.transform([processed_question]).toarray()
    prediction = model.predict(vectorized_question)
    answer_index = np.argmax(prediction)
    confidence = np.max(prediction)

    if confidence < 0.5:
        return "I'm not sure how to respond to that. Can you rephrase?"

    return label_encoder.inverse_transform([answer_index])[0]

# Cyberbullying classification
def predict_label(text):
    prediction = pipeline.predict([text])
    return prediction[0]

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_text = None
    prediction_description = None

    if request.method == 'POST':
        user_input = request.form['text_input']
        predicted_label = predict_label(user_input)

        if predicted_label == 1:
            prediction_text = "Cyberbullying"
            prediction_description = (
                "Cyberbullying involves online harassment, threats, or abusive messages intended to harm, "
                "intimidate, or degrade someone."
            )
        elif predicted_label == 0:
            prediction_text = "Non-Cyberbullying"
            prediction_description = (
                "Non-Cyberbullying refers to normal, respectful, and friendly online communication."
            )

    return render_template('predict.html', prediction_text=prediction_text, prediction_description=prediction_description)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json.get("message", "")
    bot_response = get_answer(user_message)
    return jsonify({"response": bot_response})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

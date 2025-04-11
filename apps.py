from flask import Flask, render_template, request
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained pipeline
pipeline = joblib.load('model_pipeline.joblib')

# Function to predict the label of a new input
def predict_label(text):
    prediction = pipeline.predict([text])
    return prediction[0]

# Route for homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

# Route for about page
@app.route('/about')
def about():
    return render_template('about.html')

# Route for predict page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_text = None
    prediction_description = None  # New variable for description

    if request.method == 'POST':
        # Get the input text from the form
        user_input = request.form['text_input']

        # Get the prediction
        predicted_label = predict_label(user_input)

        # Determine the human-readable label and description
        if predicted_label == 1:
            prediction_text = "Cyberbullying"
            prediction_description = (
                "Cyberbullying involves online harassment, threats, or abusive messages intended to harm, "
                "intimidate, or degrade someone. It includes offensive language, insults, and aggressive behavior."
            )
        elif predicted_label == 0:
            prediction_text = "Non-Cyberbullying"
            prediction_description = (
                "Non-Cyberbullying refers to normal, respectful, and friendly online communication. "
                "It includes constructive discussions, positive messages, and supportive content."
            )

    return render_template('predict.html', prediction_text=prediction_text, prediction_description=prediction_description)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

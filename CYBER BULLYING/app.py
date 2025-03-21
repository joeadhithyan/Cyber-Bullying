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
    if request.method == 'POST':
        # Get the input text from the form
        user_input = request.form['text_input']

        # Get the prediction
        predicted_label = predict_label(user_input)

        # Determine the human-readable label based on the prediction
        if predicted_label == 1:
            prediction_text = "Cyberbullying"
        elif predicted_label == 0:
            prediction_text = "Non-Cyberbullying"

    return render_template('predict.html', prediction_text=prediction_text)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

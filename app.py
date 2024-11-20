from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model_path = "best_model.pkl"
try:
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
except Exception as e:
    print(f"Error loading model: {str(e)}")

# Define the home route
@app.route('/')
def index():
    return render_template("index.html")

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        form_data = request.form

        # Extract input features in the same order used during model training
        features = [
            float(form_data["Glucose"]),
            float(form_data["Cholesterol"]),
            float(form_data["Hemoglobin"]),
            float(form_data["Platelets"]),
            float(form_data["WhiteBloodCells"]),
            float(form_data["RedBloodCells"]),
            float(form_data["Hematocrit"]),
            float(form_data["MeanCorpuscularVolume"]),
            float(form_data["MeanCorpuscularHemoglobin"]),
            float(form_data["MeanCorpuscularHemoglobinConcentration"]),
            float(form_data["Insulin"]),
            float(form_data["BMI"]),
            float(form_data["SystolicBloodPressure"]),
            float(form_data["DiastolicBloodPressure"]),
            float(form_data["Triglycerides"]),
            float(form_data["HbA1c"]),
            float(form_data["LDLCholesterol"]),
            float(form_data["HDLCholesterol"]),
            float(form_data["ALT"]),
            float(form_data["AST"]),
            float(form_data["HeartRate"]),
            float(form_data["Creatinine"]),
            float(form_data["Troponin"]),
            float(form_data["CReactiveProtein"])
        ]

        # Convert the input features into a numpy array
        input_features = np.array([features])

        # Predict probability
        probabilities = model.predict_proba(input_features)[0]
        ckd_positive_prob = probabilities[1]
        ckd_negative_prob = probabilities[0]

        # Adjust the threshold if needed
        threshold = 0.4  # Adjust based on testing
        prediction = 1 if ckd_positive_prob > threshold else 0

        # Map prediction to a readable label
        result = "CKD Positive" if prediction == 1 else "CKD Negative"

        # Pass probabilities and result to the template for better debugging
        return render_template(
            "index.html",
            prediction=result,
            ckd_positive_prob=round(ckd_positive_prob * 100, 2),
            ckd_negative_prob=round(ckd_negative_prob * 100, 2),
            form_data=form_data
        )

    except Exception as e:
        # Handle errors and display them to the user
        error_message = f"An error occurred: {str(e)}"
        return render_template("index.html", error=error_message)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

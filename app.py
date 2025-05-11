from flask import Flask, render_template, request
from check import predict_fraud

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    # Render the input form page
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        transaction_date = request.form["transaction_date"]
        transaction_amount = float(request.form["transaction_amount"])
        receiver_upi_id = request.form["receiver_upi_id"]
        transaction_id = request.form["transaction_id"]
        utr_number = request.form["utr_number"]

        # Combine the inputs into a feature list for prediction
        # Example: Replace this with your actual feature extraction logic
        input_data = [transaction_amount]  # Example: Only using transaction_amount for now

        # Debugging: Print input data
        print(f"Input data for prediction: {input_data}")

        # Call your prediction logic
        result, confidence = predict_fraud(input_data)

        # Debugging: Print result and confidence
        print(f"Prediction result: {result}, Confidence: {confidence}")

        # Render the result page with the prediction
        return render_template("result.html", result=result, confidence=confidence)
    except Exception as e:
        # Handle errors and return to the input form with an error message
        print(f"Error during prediction: {e}")
        return render_template("index.html", error="An error occurred. Please check your input.")

if __name__ == "__main__":
    app.run(debug=True)
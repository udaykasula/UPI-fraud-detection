# check.py
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from datetime import datetime

# Load models and encoders
rf_model = pickle.load(open('random_forest_model.pkl', 'rb'))
le_receiver = pickle.load(open('le_receiver_upi.pkl', 'rb'))
le_txn = pickle.load(open('le_transaction_id.pkl', 'rb'))
le_utr = pickle.load(open('le_utr_number.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
dl_model = load_model('deep_learning_model.h5')

# Load the pre-trained scaler and model
try:
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except (EOFError, FileNotFoundError) as e:
    print(f"Error loading files: {e}")
    exit()

def valid_transaction_id(txn_id):
    return isinstance(txn_id, str) and len(txn_id) == 23 and txn_id[0] == 'T' and txn_id[1:].isdigit()

def valid_utr_number(utr):
    return isinstance(utr, str) and len(utr) == 12 and utr.isdigit()

def preprocess_input(transaction_date, transaction_amount, receiver_upi_id, transaction_id, utr_number):
    # Validate inputs
    if not valid_transaction_id(transaction_id):
        raise ValueError("Invalid Transaction ID format. Must start with 'T' followed by 22 digits.")
    if not valid_utr_number(utr_number):
        raise ValueError("Invalid UTR Number format. Must be exactly 12 digits.")

    try:
        date_ordinal = datetime.strptime(transaction_date, '%Y-%m-%d').toordinal()
    except Exception:
        raise ValueError("Invalid date format. Use YYYY-MM-DD.")

    # Encode categorical features with unknown handling
    def encode_with_unknown(le, val):
        if val in le.classes_:
            return le.transform([val])[0]
        else:
            return len(le.classes_)  # unknown category

    receiver_enc = encode_with_unknown(le_receiver, receiver_upi_id)
    txn_enc = encode_with_unknown(le_txn, transaction_id)
    utr_enc = encode_with_unknown(le_utr, utr_number)

    features = np.array([[date_ordinal, float(transaction_amount), receiver_enc, txn_enc, utr_enc]])
    features_scaled = scaler.transform(features)
    return features_scaled

def predict_fraud(input_data):
    """
    Predicts whether the input data indicates fraud or not and returns the result with accuracy percentage.
    """
    try:
        # Ensure input_data is a list or array
        if not isinstance(input_data, (list, tuple)):
            raise ValueError("Input data must be a list or tuple.")

        # Debugging: Print input data
        print(f"Input data received: {input_data}")

        # Scale the input data
        input_data_scaled = scaler.transform([input_data])  # Ensure input_data is a list or array
        print(f"Scaled input data: {input_data_scaled}")

        # Predict probabilities
        probabilities = model.predict_proba(input_data_scaled)[0]  # Get probabilities for each class
        print(f"Prediction probabilities: {probabilities}")

        fraud_probability = probabilities[1]  # Assuming class 1 is "Fraud"

        # Determine the result
        if fraud_probability > 0.5:
            result = "Fraud"
        else:
            result = "Not Fraud"

        # Calculate confidence percentage
        confidence = fraud_probability * 100 if result == "Fraud" else (1 - fraud_probability) * 100

        return result, round(confidence, 2)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error", 0.0
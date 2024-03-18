from flask import Flask, request, jsonify
import traceback
from joblib import load
import logging
import os
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Load the trained pipeline
try:
    pipeline_path = os.path.join(os.getcwd(), 'mlu', 'deployment', 'trained_pipeline.joblib')  # Updated path to reflect modular structure
    pipeline = load(pipeline_path)
    logging.info("Trained pipeline loaded successfully.")
except FileNotFoundError as e:
    logging.error(f"Trained pipeline file not found: {traceback.format_exc()}")
    raise Exception(f"Trained pipeline file not found at {pipeline_path}. Ensure the file exists.")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            json_data = request.json
            # Basic input validation
            if not json_data or "data" not in json_data:
                logging.error("Missing 'data' field in request JSON.")
                return jsonify({"error": "Missing 'data' field in request JSON."}), 400
            
            # Ensure data is in the correct format for prediction
            data = json_data["data"]
            if isinstance(data, list):
                # If data is a list of records, convert to numpy array
                data = np.array(data)
            elif isinstance(data, dict):
                # If data is a single record, wrap it in a list then convert to numpy array
                data = np.array([list(data.values())])
            else:
                return jsonify({"error": "Data format not recognized. Please provide a list of records or a single record as a dictionary."}), 400

            logging.info("Prediction request received with data.")
            prediction = pipeline.predict(data)
            logging.info("Prediction made successfully.")
            return jsonify({"prediction": prediction.tolist()})
        except Exception as e:
            logging.error(f"Error during prediction: {traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = os.getenv('FLASK_PORT', '8001')  # Changed default port to 8001 to avoid conflict with reserved ports
    app.run(host='0.0.0.0', port=int(port), debug=True)
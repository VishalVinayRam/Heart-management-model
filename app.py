from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained LSTM model
model = load_model('lstm_model.h5')

# Define an endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Assuming you send the input data as JSON
    input_sequence = np.array(data['sequence'])
    input_sequence = np.expand_dims(input_sequence, axis=0)  # Add batch dimension

    # Make a prediction
    raw_prediction = model.predict(input_sequence)
    probability = raw_prediction[0][0]  # Extract the probability value from the output

# Convert numpy float32 to native Python float
    probability = float(probability)

# Return the probability as JSON
    return jsonify({'probability': probability})
    # Return the prediction as JSON

if __name__ == '__main__':
    app.run(debug=True)

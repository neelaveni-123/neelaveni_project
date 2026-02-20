from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and scaler from the current directory
model = pickle.load(open('model.save', 'rb'))
scaler = pickle.load(open('transform.save', 'rb'))

@app.route('/')
def home():
    # UPDATED: Now points to your new high-end template
    return render_template('Sensorpredict.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data from the 7 input fields
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Scale and predict
    scaled_features = scaler.transform(final_features)
    prediction = model.predict(scaled_features)
    
    # UPDATED: Returns result to the new template
    return render_template('Sensorpredict.html', 
                          prediction_text=f'Predicted Rotor Temp: {prediction[0]:.2f}°C')

if __name__ == "__main__":
    app.run(debug=True)
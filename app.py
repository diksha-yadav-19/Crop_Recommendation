from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model and scalers
model = pickle.load(open('model.pkl', 'rb')) 
sc = pickle.load(open('standscaler.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

@app.route('/')
def index():
    # Render the main page
    return render_template('index.html')

@app.route('/recommend')
def recommendation_form():
    # Render the recommendation form page
    return render_template('recommandation.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            # Extract form data
            N = float(request.form['Nitrogen'])
            P = float(request.form['Phosphorus'])
            K = float(request.form['Potassium'])
            temp = float(request.form['Temperature'])
            humidity = float(request.form['Humidity'])
            ph = float(request.form['pH'])
            rainfall = float(request.form['Rainfall'])

            # Input validation
            if not (0 <= N <= 200 and 0 <= P <= 200 and 0 <= K <= 200 and 0 <= temp <= 50 and 0 <= humidity <= 100 and 0 <= ph <= 14 and 0 <= rainfall <= 10000):
                return jsonify({"error": "Invalid input values. Please check the ranges."})

            # Prepare the feature array for prediction
            feature_list = [N, P, K, temp, humidity, ph, rainfall]
            single_pred = np.array(feature_list).reshape(1, -1)

            # Scale features
            mx_features = mx.transform(single_pred)
            sc_mx_features = sc.transform(mx_features)

            # Make prediction
            prediction = model.predict(sc_mx_features)

            # Define crop dictionary
            crop_dict = {
                1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
            }

            # Get the crop name from the prediction
            crop = crop_dict.get(prediction[0], "Unknown crop")

            # Return the result as JSON
            return jsonify({'result': crop})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=8000)






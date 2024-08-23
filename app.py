from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/car_price_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        name = request.form.get('name')
        year = int(request.form.get('year'))
        km_driven = int(request.form.get('km_driven'))
        fuel = request.form.get('fuel')
        seller_type = request.form.get('seller_type')
        transmission = request.form.get('transmission')
        owner = request.form.get('owner')
        
        # Process input data
        data = pd.DataFrame([[year, km_driven, fuel, seller_type, transmission, owner]],
                            columns=['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner'])
        
        # One-hot encoding for categorical features
        data = pd.get_dummies(data, columns=['fuel', 'seller_type', 'transmission', 'owner'])
        
        # Align data with training features
        data = data.reindex(columns=model.feature_names_in_, fill_value=0)
        
        # Make prediction
        prediction = model.predict(data)[0]
        
        # Render the prediction on the index page
        return render_template('index.html', prediction_text=f'Estimated Selling Price: ₹{prediction:.2f}')
    
    except Exception as e:
        return f"An error occurred: {str(e)}", 400

if __name__ == "__main__":
    app.run(debug=True)

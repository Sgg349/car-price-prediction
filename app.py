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
        
        # Process input data into a DataFrame
        data = pd.DataFrame([[year, km_driven, fuel, seller_type, transmission, owner]],
                            columns=['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner'])
        
        # Align data with model features
        data_transformed = model.named_steps['preprocessor'].transform(data)
        
        # Convert the transformed data back to a DataFrame
        data_transformed_df = pd.DataFrame(data_transformed, columns=model.named_steps['preprocessor'].get_feature_names_out())
        
        # Make prediction
        prediction = model.named_steps['regressor'].predict(data_transformed_df)[0]
        
        # Render the prediction on the index page
        return render_template('index.html', prediction_text=f'Estimated Selling Price: ₹{prediction:.2f}')
    
    except Exception as e:
        return f"An error occurred: {str(e)}", 400

if __name__ == "__main__":
    app.run(debug=True)

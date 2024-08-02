from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model dan scaler
model_rf = joblib.load('model_rf.pkl')
fiture_scalar = joblib.load('feature_scalar.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data dari form
    data = request.form.to_dict()
    df_new_data = pd.DataFrame([data])
    
    # Konversi tipe data sesuai kebutuhan
    df_new_data = df_new_data.astype({
        'Size (cm)': float,
        'Weight (g)': float,
        'Brix (Sweetness)': float,
        'pH (Acidity)': float,
        'Softness (1-5)': float,
        'HarvestTime (days)': int,
        'Ripeness (1-5)': float,
        'Color': int,
        'Blemishes (Y/N)': int
    })

    # Menghitung densitas
    df_new_data['Density (g/cm³)'] = df_new_data['Weight (g)'] / (df_new_data['Size (cm)'] ** 3)

    # Memisahkan kolom untuk scaling
    Blemishes_column = df_new_data['Blemishes (Y/N)']
    Softness_column = df_new_data['Softness (1-5)']
    Ripeness_column = df_new_data['Ripeness (1-5)']
    Color_column = df_new_data['Color']
    df_new_data = df_new_data.drop(['Blemishes (Y/N)', 'Softness (1-5)', 'Ripeness (1-5)', 'Color'], axis=1)

    # Melakukan scaling pada kolom yang sesuai
    new_features_scaled = fiture_scalar.transform(df_new_data)

    # Membuat DataFrame hasil scaling
    new_features_scaled = pd.DataFrame(new_features_scaled, columns=df_new_data.columns)

    # Menambahkan kembali kolom yang tidak di-scaling
    new_features_scaled['Blemishes (Y/N)'] = Blemishes_column.values
    new_features_scaled['Softness (1-5)'] = Softness_column.values
    new_features_scaled['Ripeness (1-5)'] = Ripeness_column.values
    new_features_scaled['Color'] = Color_column.values

    # Melakukan prediksi
    prediction = model_rf.predict(new_features_scaled)
    prediction += 1  # Adjusting prediction to match original scale

    return render_template('index.html', prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)

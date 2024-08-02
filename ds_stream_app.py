import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and scaler
model_rf = joblib.load('model_rf.pkl')
feature_scaler = joblib.load('feature_scalar.pkl')

# Streamlit application
st.title('Predict Quality of Fruit')

# Input fields
size = st.number_input('Size (cm)', min_value=0.1, step=0.1)
weight = st.number_input('Weight (g)', min_value=0.1, step=0.1)
brix = st.number_input('Brix (Sweetness)', min_value=0.0, step=0.1)
ph = st.number_input('pH (Acidity)', min_value=0.0, step=0.1)
softness = st.number_input('Softness (1-5)', min_value=1.0, max_value=5.0, step=0.5)
harvest_time = st.number_input('Harvest Time (days)', min_value=0, step=1)
ripeness = st.number_input('Ripeness (1-5)', min_value=1.0, max_value=5.0, step=0.5)
color = st.selectbox('Color', ['Deep Orange', 'Light Orange', 'Orange-Red', 'Orange', 'Yellow-Orange'])
blemishes = st.selectbox('Blemishes (Y/N)', ['No', 'Yes'])

# Convert inputs to appropriate format
color_map = {'Deep Orange': 0, 'Light Orange': 1, 'Orange-Red': 2, 'Orange': 3, 'Yellow-Orange': 4}
blemishes_map = {'No': 0, 'Yes': 1}

input_data = {
    'Size (cm)': size,
    'Weight (g)': weight,
    'Brix (Sweetness)': brix,
    'pH (Acidity)': ph,
    'Softness (1-5)': softness,
    'HarvestTime (days)': harvest_time,
    'Ripeness (1-5)': ripeness,
    'Color': color_map[color],
    'Blemishes (Y/N)': blemishes_map[blemishes]
}

df_new_data = pd.DataFrame([input_data])

# Calculate density
df_new_data['Density (g/cm³)'] = df_new_data['Weight (g)'] / (df_new_data['Size (cm)'] ** 3)

# Check for inf or NaN values
if not np.isfinite(df_new_data['Density (g/cm³)'].values).all():
    st.error("Invalid input: Density calculation resulted in infinity or NaN values.")
else:
    # Separate columns for scaling
    Blemishes_column = df_new_data['Blemishes (Y/N)']
    Softness_column = df_new_data['Softness (1-5)']
    Ripeness_column = df_new_data['Ripeness (1-5)']
    Color_column = df_new_data['Color']
    df_new_data = df_new_data.drop(['Blemishes (Y/N)', 'Softness (1-5)', 'Ripeness (1-5)', 'Color'], axis=1)

    # Scale features
    new_features_scaled = feature_scaler.transform(df_new_data)
    new_features_scaled = pd.DataFrame(new_features_scaled, columns=df_new_data.columns)

    # Add back non-scaled columns
    new_features_scaled['Blemishes (Y/N)'] = Blemishes_column.values
    new_features_scaled['Softness (1-5)'] = Softness_column.values
    new_features_scaled['Ripeness (1-5)'] = Ripeness_column.values
    new_features_scaled['Color'] = Color_column.values

    # Prediction
    if st.button('Predict'):
        prediction = model_rf.predict(new_features_scaled)
        prediction += 1  # Adjusting prediction to match original scale
        st.write(f'Predicted Quality: {prediction[0]}')
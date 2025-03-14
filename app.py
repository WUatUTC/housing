import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
import joblib
from keras.saving import register_keras_serializable
import tensorflow as tf

# Register the custom mse function
@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Load the saved ANN model and preprocessor
custom_objects = {'mse': mse}
model = keras.models.load_model('ann_model.h5', custom_objects=custom_objects)
preprocessor = joblib.load('preprocessor.pkl')

st.title('AI Powered Ames Housing Price Predictor')
st.write("This app was powered by a trained AI model. Please enter essential house details below to predict the sale price:")

# -- User Inputs for the selected features --

# Since 'Age' in our data is computed as [Yr Sold] - [Year Built],
# we ask for Age directly (you might also choose to compute it if you collect both)
age = st.number_input('Age (years)', min_value=0, value=30)

gr_liv_area = st.number_input('Gr Liv Area (sq ft)', min_value=0, value=1500)
lot_area = st.number_input('Lot Area (sq ft)', min_value=0, value=5000)
overall_qual = st.slider('Overall Qual (1-10)', min_value=1, max_value=10, value=5)

# For the categorical Neighborhood, define a sample list or load from your dataset
neighborhood_options = ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel']
neighborhood = st.selectbox('Neighborhood', options=neighborhood_options)

# When the user clicks the button, process input and predict
if st.button('Predict Sale Price'):
    # Create a dataframe matching the expected input structure
    input_data = pd.DataFrame({
        'Age': [age],
        'Gr Liv Area': [gr_liv_area],
        'Lot Area': [lot_area],
        'Overall Qual': [overall_qual],
        'Neighborhood': [neighborhood]
    })

    # Preprocess the input data using the saved preprocessor
    input_processed = preprocessor.transform(input_data)
    if hasattr(input_processed, "toarray"):
        input_processed = input_processed.toarray()

    # Predict the sale price using the loaded model
    prediction = model.predict(input_processed)
    predicted_price = prediction[0][0]

    st.success(f"Predicted Sale Price: ${predicted_price:,.0f}")


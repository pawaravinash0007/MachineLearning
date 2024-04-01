import streamlit as st
import pickle
import numpy as np

# Load the trained model from the .sav file
model = pickle.load(open('avinash_model.sav', 'rb'))

# Streamlit app UI
st.title('Model Deployment with Streamlit')
st.header('Predictions using a Trained Model')

# User input
sepal_length = st.slider('Sepal Length (cm)', 0.0, 10.0, 5.0)
sepal_width = st.slider('Sepal Width (cm)', 0.0, 10.0, 3.0)
petal_length = st.slider('Petal Length (cm)', 0.0, 10.0, 2.0)
petal_width = st.slider('Petal Width (cm)', 0.0, 10.0, 1.0)

# Make predictions
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]

# Display prediction
st.write(f'Predicted Class: {prediction}')

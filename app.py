import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
clf = pickle.load(open("avinash_model.pkl","rb"))

def predict(data):
    clf = pickle.load(open("avinash_model.pkl","rb"))
    return clf.predict(data)
    
st.title('Classifying Iris Flowers')
st.markdown("This model is design at NIELIT Daman Smart Lab ")

st.header("Plant Features")
# Define the sliders for input
sepal_l = float(st.slider('Sepal length (cm)', 1.0, 8.0, 0.5))
sepal_w = float(st.slider('Sepal width (cm)', 2.0, 4.4, 0.5))
petal_l = float(st.slider('Petal length (cm)', 1.0, 7.0, 0.5))
petal_w = float(st.slider('Petal width (cm)', 0.1, 2.5, 0.5))

# Predict and display the result when the button is clicked
if st.button("Predict type of Iris"):
    result = clf.predict(np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    st.text(result[0])

st.markdown('Developed By- Exeternal Guide : Avinash Pawar and WBL Intern Team')

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# Load the saved model
model = joblib.load('iris_logreg_model.pkl')

# Streamlit interface
st.title('Iris Flower Classification')

# Create input fields for the user to input the features of the iris flower
sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.0)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 4.0)
petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.0)

# Create a prediction button
if st.button('Predict'):
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make prediction
    prediction = model.predict(input_features)
    
    # Map the prediction to the flower name
    iris = load_iris()
    iris_target_names = iris.target_names
    flower_name = iris_target_names[prediction][0]
    
    # Show the prediction
    st.write(f'The predicted flower type is: **{flower_name}**')

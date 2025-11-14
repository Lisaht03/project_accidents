import streamlit as st
import requests

st.title ('Predicting Severe Road Accidents in Île-de-France')

st.write('Select features below:')

sep_len = st.slider('Select a value for Sepal length', min_value=0, max_value=10, value=1, step=1)
sep_wid = st.slider('Select a value for Sepal width', min_value=0, max_value=10, value=1, step=1)
pet_len = st.slider('Select a value for Petal length', min_value=0, max_value=10, value=1, step=1)
pet_wid = st.slider('Select a value for Petal width', min_value=0, max_value=10, value=1, step=1)

url='https://testapi-114787831451.europe-west1.run.app/predict'

params={
    'sepal_length': sep_len,
    'sepal_width': sep_wid,
    'petal_length': pet_len,
    'petal_width': pet_wid
}

try:
    response = requests.get(url=url, params=params, timeout=5)
    result = response.json()
    st.write('The flower belongs to class:', result)
except Exception as e:
    st.error(f'Error to connect with API or parse response: {e}')

classes = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
predicted_class = classes.get(result['flower'], "Unknown")
st.success(f"The flower belongs to class: {predicted_class}")

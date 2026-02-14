import pickle
import pandas as pd
import numpy as np
import streamlit as st


def predict_species(sep_len, sep_width, pet_len, pet_width, scaler_path, model_path):
    try:
        with open(scaler_path, 'rb') as file1:
            scaler = pickle.load(file1)

        with open(model_path, 'rb') as file2:
            model = pickle.load(file2)

        dct = {
            'sepallength': [sep_len],
            'sepalwidth': [sep_width],
            'petallength': [pet_len],
            'petalwidth': [pet_width]
        }

        x_new = pd.DataFrame(dct)
        xnew_pre = scaler.transform(x_new)

        pred = model.predict(xnew_pre)
        prob = model.predict_proba(xnew_pre)
        max_prob = np.max(prob)

        return pred[0], max_prob

    except Exception as e:
        st.error(f'Error during prediction: {str(e)}')
        return None, None


# Streamlit UI
st.title("🌸 Iris Species Prediction")

sep_len = st.number_input('Sepal Length', min_value=0.0, step=0.1, value=5.1)
sep_width = st.number_input('Sepal Width', min_value=0.0, step=0.1, value=3.5)
pet_len = st.number_input('Petal Length', min_value=0._

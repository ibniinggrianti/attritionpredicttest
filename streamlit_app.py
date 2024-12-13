import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Attrition Prediction')

st.subheader("Is your job worth keeping? Should you stay? Or just leave? Let's try!")
st.write("You can see below for more information")

# Load dataset (Ensure the CSV file is in the correct location)
df = pd.read_csv("https://raw.githubusercontent.com/ibniinggrianti/attritionprediction/refs/heads/master/IBM-HR-Analytics-Employee-Attrition-and-Performance-Revised.csv")

X = df.drop('Attrition', axis=1)

y = df.Attrition
  
with st.sidebar:
  st.header('Input Features')
  Age = st.slider("Age", min_value=18, max_value=60, value=25)
  st.write(f"Your selected option: {Age}.")

# DataFrame for the input features
data = {
    'Age': [Age],
}
input_df = pd.DataFrame(data)
input_attrition = pd.concat([input_df, X], axis=0)

# Display in Streamlit
with st.expander('Input Features'):
    st.write('**Input Features:**')
    st.dataframe(input_df)

    st.write('**Combined Attrition Data (Input + Original Dataset):**')
    st.dataframe(input_attrition)

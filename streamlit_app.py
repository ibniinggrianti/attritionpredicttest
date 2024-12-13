import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title('Attrition Prediction')

st.subheader("Is your job worth keeping? Should you stay? Or just leave? Let's try!")
st.write("You can see below for more information")

# Load dataset (Ensure the CSV file is in the correct location)
df = pd.read_csv("https://raw.githubusercontent.com/ibniinggrianti/attritionpredicttest/refs/heads/master/editedIBM-HR-Analytics-Employee-Attrition-and-Performance-Revised.csv")

X_raw = df.drop('Attrition', axis=1)

y_raw = df['Attrition']

with st.sidebar:
  st.header('Input Features')
  Age = st.slider("Age", min_value=18, max_value=60, value=25)
  st.write(f"Your selected age: {Age}.")

  Gender_options = ["Male", "Female"]
  selected_gender = st.pills("Gender", Gender_options, selection_mode="single")
  #Gender = selected_gender[0] if selected_gender else None  # Handle empty selection
  st.markdown(f"Your selected gender: {selected_gender}.")

  MaritalStatus_options = ["Single", "Married", "Divorced"]
  selected_marital_status = st.pills("Marital Status", MaritalStatus_options, selection_mode="single")
  #MaritalStatus = selected_marital_status[0] if selected_marital_status else None  # Handle empty selection
  st.markdown(f"Your selected Marital Status: {selected_marital_status}.")

# DataFrame for the input features
data = {
    'Age': [Age],
    'Gender': [Gender],
    'MaritalStatus': [MaritalStatus]
}
#data['Gender'] = pd.Series(data['Gender']).map({'F': 'Female', 'M': 'Male'})
#data['MaritalStatus'] = pd.Series(data['MaritalStatus']).map({'S': 'Single', 'M': 'Married', 'D': 'Divorced'})
input_df = pd.DataFrame(data, index=[0])
input_attrition = pd.concat([input_df, X_raw], axis=0)

# Display in Streamlit
with st.expander('Input Features'):
    st.write('**Input Features:**')
    st.dataframe(input_df)

    st.write('**Combined Attrition Data (Input + Original Dataset):**')
    st.dataframe(input_attrition)
  
#Encode
encode = ['Gender', 'MaritalStatus']
df_attrition = pd.get_dummies(input_attrition, columns=encode, prefix=encode)
#df_attrition[:1]
  
X = df_attrition[1:]
input_row = df_attrition[:1]

# Encode y
target_mapper = {'Yes': 0,
                 'No': 1,}
#def target_encode(val):
  #return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data Preparation'):
  st.write('**Encoded X (Input Features)**')
  st.dataframe(input_row)
  st.write('**Encoded y**')
  st.dataframe(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

# Display results in Streamlit
with st.expander('Model Prediction'):
    st.write(f"**Prediction:** {'Attrition' if prediction[0] == 1 else 'No Attrition'}")
    st.write("**Prediction Probabilities:**")
    st.dataframe(pd.DataFrame(prediction_proba, columns=['No Attrition', 'Attrition']))


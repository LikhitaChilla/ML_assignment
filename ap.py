import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# Load the pickled model
with open('poly_regression_modell.pkl', 'rb') as model_file:
    model,feature_names = pickle.load(model_file)
with open('label_encoder.pkl', 'rb') as model_file:
    encoder = pickle.load(model_file)


# Function to make predictions
def predict(input_data):
    poly = PolynomialFeatures(degree=2)
    x_train_poly = poly.fit_transform(input_data)
    y_pred = model.predict(x_train_poly)
    return y_pred


# Streamlit app
st.title('Model Deployment with Streamlit')
# Input form
st.sidebar.header('Input Parameters')
param1 = st.sidebar.number_input('Hours_Studied', value=0)
param2 = st.sidebar.number_input('Attendance', value=0)
param6 = st.sidebar.number_input('Sleep_Hours', value=0)
param7 = st.sidebar.number_input('Previous_Scores', value=0)
param10 = st.sidebar.number_input('Tutoring_Sessions', value=0)
param15 = st.sidebar.number_input('Physical_Activity',value=0)
features={
    'Parental_Involvement':['Low','Medium','High'],
    'Access_to_Resources':['Low','Medium','High'],
    'Extracurricular_Activities':['No','Yes'],
    'Motivation_Level':['Low','Medium','High'],
    'Internet_Access':['No','Yes'],
    'Family_Income':['Low','Medium','High'],
    'Teacher_Quality':['Low','Medium','High'],
    'School_Type':['Public','Private'],
    'Peer_Influence':['Positive','Negative','Neutral'],
    'Learning_Disabilities':['No','Yes'],
    'Parental_Education_Level':['High School','College','Postgraduate'],
    'Distance_from_Home':['Near','Moderate','Far'],
    'Gender':['Male','Female']
}

selected_features={}
for feature,options in features.items():
    selected_features[feature]=st.sidebar.selectbox(f'Select {feature}',options)


    
data = {
    'Hours_Studied': param1,
    'Attendance': param2,
    'Parental_Involvement': selected_features['Parental_Involvement'],
    'Access_to_Resources': selected_features['Access_to_Resources'],
    'Extracurricular_Activities': selected_features['Extracurricular_Activities'],
    'Sleep_Hours': param6,
    'Previous_Scores': param7,
    'Motivation_Level': selected_features['Motivation_Level'],
    'Internet_Access': selected_features['Internet_Access'],
    'Tutoring_Sessions': param10,
    'Physical_Activity': param15,
    'Family_Income': selected_features['Family_Income'],
    'Teacher_Quality': selected_features['Teacher_Quality'],
    'School_Type': selected_features['School_Type'],
    'Peer_Influence': selected_features['Peer_Influence'],
    'Learning_Disabilities': selected_features['Learning_Disabilities'],
    'Parental_Education_Level': selected_features['Parental_Education_Level'],
    'Distance_from_Home': selected_features['Distance_from_Home'],
    'Gender': selected_features['Gender']
}
def load_and_transform(feature, new_value):
    # Load the corresponding pickled encoder
    with open(f'{feature}_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    
    # Transform the new user input
    transformed_value = encoder.transform(new_value)
    
    return transformed_value

encoded_user_data = {}
for feature, value in selected_features.items():
    data[feature] = load_and_transform(feature,[value])

print("Encoded new user data:", encoded_user_data)

df=pd.DataFrame([data],columns=['Hours_Studied','Attendance','Parental_Involvement','Access_to_Resources','Extracurricular_Activities','Sleep_Hours','Previous_Scores','Motivation_Level','Internet_Access','Tutoring_Sessions','Physical_Activity','Family_Income',
                       'Teacher_Quality','School_Type','Peer_Influence','Learning_Disabilities','Parental_Education_Level','Distance_from_Home','Gender'
                       ])



# cat_col=[]
# for i in df.columns:
#     if df[i].dtypes=='object':
#         cat_col.append(i)
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# for col in cat_col:
#     df[col] = encoder.transform(df[col])
df=df[feature_names]
input_data =df
# Make prediction on button click
if st.button('Predict'):
    prediction = predict(input_data)
    dd=pd.DataFrame(list(selected_features.items()),columns=['feature','selected value'])
    st.table(dd)
    st.write(f'The predicted student performance is : {prediction[0]}')

# Optional: Display the model's details
if st.checkbox('Show Model Info'):
    st.write(model)

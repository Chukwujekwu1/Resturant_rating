import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

scaler = joblib.load("Scaler.pkl")

st.set_page_config(layout="wide")

st.title("Resturan Rating Preprocessing App")

st.caption("This app helps you predict a resturant review class")

st.divider()

# inputs that our models takes
averagecost = st.number_input("Please enter the estimated averge cost for two", min_value=50,max_value=999999,value=1000,step=200)

tablebooking = st.selectbox("Resturant has table booking?",["Yes","No"])

onlinedelivery = st.selectbox("Resturants has online booking?",["Yes","No"]) 

pricerange = st.selectbox("What is the price range (1 Cheapest, 4 Most Expensive)",[1,2,3,4])

predict_button = st.button("Predict the reviews!")

st.divider()

model = joblib.load("mlmodel.pkl")

bookingstatus = 1 if tablebooking == "Yes" else 0

deliverystatus = 1 if onlinedelivery == "Yes" else 0

values = [[averagecost,bookingstatus,deliverystatus,pricerange]]

my_X_values = np.array(values)

x = scaler.transform(my_X_values)

if predict_button:
    st.snow()
    
    prediction = model.predict(x)
    
    # st.write(prediction)
    
    if prediction < 2.5:
        st.write("Poor")
    elif prediction < 3.5:
        st.write("Average")
    elif prediction < 4.0:
        st.write("Good")
    elif prediction < 4.5:
        st.write("Very Good")
    else:
        st.write("Excellent")
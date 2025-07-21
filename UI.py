import streamlit as st

st.markdown('<h1 style="text-align: center;">Workout Assistant</h1>', unsafe_allow_html=True)

st.markdown("**Experience**")
experience = st.selectbox("Select your experience level", ["Beginner", "Intermediate", "Advanced"])

st.markdown("**Gender**")
gender = st.selectbox('Select your gender', ['Male', 'Female'])

st.markdown("**Height (in.)**")
height = st.text_input("Enter your height")

st.markdown("**Weight (lbs.)**")
weight = st.text_input("Enter your weight")

if st.button("Submit"):
    st.success('Processing Information')
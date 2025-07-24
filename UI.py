import streamlit as st

from models.deep_learning import final_recommendations

st.markdown('<h1 style="text-align: center;">Workout Assistant</h1>', unsafe_allow_html=True)

st.markdown("**Experience**")
experience = st.selectbox("Select your experience level", ["Beginner", "Intermediate", "Advanced"])

st.markdown("**Gender**")
gender = st.selectbox('Select your gender', ['Male', 'Female'])

st.markdown("**Height (in.)**")
height = st.text_input("Enter your height")

st.markdown("**Weight (lbs.)**")
weight = st.text_input("Enter your weight")

st.markdown("**Age**")
age = st.number_input("Enter your age", min_value=16, max_value=80, value=25)

if st.button("Submit"):
    with st.spinner("Generating workouts"):
        height_float = float(height)
        weight_float = float(weight)
            
        recommendations = final_recommendations(gender, age, height_float, weight_float, experience)
        
    if len(recommendations) > 0:
        for i, recommendation in enumerate(recommendations, 1):
            st.write(f"**Exercise {i}:** {recommendation['title']}")
            st.write(f"**Description:** {recommendation['description']}")
            st.write("---")
    else:
        st.error("Error generating workouts. Please try again.")
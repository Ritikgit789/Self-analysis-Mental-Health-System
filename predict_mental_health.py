import streamlit as st
import joblib
import numpy as np
import os
import google.generativeai as genai
from dotenv import load_dotenv
import base64




# Load the trained Random Forest model
model = joblib.load("E:/Arogo AI/mental_health_model(XGBOOST).pkl")

# Configure Gemini API
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



st.markdown(f"""
    <style>
        .main{{
            background-style: cover;
            background-position: center;
            background-repeat:no-repeat;
        }}

        /* Sidebar Styling */
        .stSidebar {{
            background-color: rgba(0, 0, 0, 0.8);
            padding: 20px;
            border-radius: 10px;
        }}

        /* Buttons */
        .stButton>button {{
            background-color: #ff6b6b;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
        }}

        /* Text Inputs */
        .stTextInput>div>div>input {{
            border-radius: 8px;
            padding: 10px;
            font-size: 16px;
        }}

        /* Titles */
        h1, h2, h3 {{
            text-align: center;
            color: #f8f9fa;
        }}
    </style>
""", unsafe_allow_html=True)

# Close the 'main' div tag after content
st.markdown('</div>', unsafe_allow_html=True)

# Define important features for prediction
important_features = [
       'Age', 'Gender', 'self_employed', 'family_history', 'treatment',
       'work_interfere', 'no_employees', 'remote_work', 'tech_company',
       'benefits', 'care_options', 'wellness_program', 'seek_help',
       'anonymity', 'leave', 'mental_health_consequence',
       'phys_health_consequence', 'coworkers', 'supervisor',
       'mental_health_interview', 'phys_health_interview',
       'mental_vs_physical', 'obs_consequence',
]

# Function to make severity predictions
def predict_severity(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    severity_map = {0: "Mild", 1: "Moderate", 2: "Severe"}
    return severity_map[prediction]

# Function to get explanation from Gemini
def get_explanation(severity):
    prompt = f"""
    A person has been predicted with '{severity}' mental health severity.
    Explain in natural language what this means and suggest personalized coping mechanisms.
    Provide steps for managing stress, improving mental well-being, and when to seek professional help.
    """
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")
    response = model_gemini.generate_content(prompt)
    return response.text


# Streamlit UI
st.title("🧠 Psycho AI - Mental Health Prediction & AI Chatbot")

# Sidebar for user input
st.sidebar.header("📝 Mental Health Assessment Tool")


# User inputs
age = st.sidebar.slider("Age", 18, 100, 30)
gender = st.sidebar.radio("Gender", [("Male", 1), ("Female", 0)])
self_employed = st.sidebar.radio("Are you self-employed?", [("Yes", 1), ("No", 0)])
family_history = st.sidebar.radio("Family History of Mental Illness?", [("Yes", 1), ("No", 0)])
treatment = st.sidebar.radio("Have you taken treatment for mental health?", [("Yes", 1), ("No", 0)])
work_interfere = st.sidebar.radio("Does Work Interfere with Mental Health?", [("Yes", 1), ("No", 0), ("Sometimes", 2)])
no_employees = st.sidebar.radio("Number of Employees at Workplace?", [("1-5", 1), ("6-25", 2), ("26-100", 3), ("100+", 4)])
remote_work = st.sidebar.radio("Do you work remotely?", [("Yes", 1), ("No", 0)])
tech_company = st.sidebar.radio("Do you work in a tech company?", [("Yes", 1), ("No", 0)])
benefits = st.sidebar.radio("Does your company provide mental health benefits?", [("Yes", 1), ("No", 0)])
care_options = st.sidebar.radio("Does your workplace offer mental health care?", [("Yes", 1), ("No", 0)])
wellness_program = st.sidebar.radio("Does your company have a wellness program?", [("Yes", 1), ("No", 0)])
seek_help = st.sidebar.radio("Is seeking mental health help encouraged at your workplace?", [("Yes", 1), ("No", 0)])
anonymity = st.sidebar.radio("Is mental health information kept anonymous at your workplace?", [("Yes", 1), ("No", 0)])
leave = st.sidebar.radio("Is taking mental health leave easy?", [("Yes", 1), ("No", 0)])
mental_health_consequence = st.sidebar.radio("Have you faced consequences for mental health issues?", [("Yes", 1), ("No", 0), ("Sometimes", 2)])
phys_health_consequence = st.sidebar.radio("Have you faced consequences for physical health issues?", [("Yes", 1), ("No", 0)])
coworkers = st.sidebar.radio("Can you discuss mental health with coworkers?", [("Yes", 1), ("No", 0)])
supervisor = st.sidebar.radio("Can you talk to your supervisor about mental health?", [("Yes", 1), ("No", 0)])
mental_health_interview = st.sidebar.radio("Would you bring up mental health in a job interview?", [("Yes", 1), ("No", 0)])
phys_health_interview = st.sidebar.radio("Would you bring up physical health in a job interview?", [("Yes", 1), ("No", 0)])
mental_vs_physical = st.sidebar.radio("Is mental health treated the same as physical health at your workplace?", [("Yes", 1), ("No", 0)])
obs_consequence = st.sidebar.radio("Do you think discussing mental health at work could have negative consequences?", [("Yes", 1), ("No", 0)])

# Prepare input data for model
input_features = [
    age, gender[1], self_employed[1], family_history[1], treatment[1], work_interfere[1],
    no_employees[1], remote_work[1], tech_company[1], benefits[1], care_options[1],
    wellness_program[1], seek_help[1], anonymity[1], leave[1], mental_health_consequence[1],
    phys_health_consequence[1], coworkers[1], supervisor[1], mental_health_interview[1],
    phys_health_interview[1], mental_vs_physical[1], obs_consequence[1]
]


if st.sidebar.button("Predict Severity"):
    severity = predict_severity(input_features)
    st.sidebar.success(f"Predicted Mental Health Severity: **{severity}**")

    # Get explanation from Gemini
    explanation = get_explanation(severity)
    
    st.subheader("📌 Explanation & Coping Mechanisms")
    
    if explanation:
        st.write(explanation)
    else:
        st.warning("⚠️ No explanation generated. Please try again.")


# Chatbot Section
st.subheader("💬 AI Mental Health Chatbot")
user_question = st.text_input("Ask me anything about mental health:")
if st.button("Get Response"):
    if user_question:
        response = get_explanation(user_question)
        st.write("🤖 AI Chatbot:", response)
    else:
        st.warning("Please enter a question!")

# Footer
st.markdown("🔹 Powered by ML Model & Gemini AI")

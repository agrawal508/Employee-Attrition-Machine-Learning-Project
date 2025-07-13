import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load model and feature list
model = joblib.load("final_model.joblib")
features = joblib.load("feature_list.joblib")

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="🧑‍💼", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background-color: #f8f9fa;
            color: #333;
        }
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .slider-label {
            font-weight: bold;
            margin-top: -10px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="gradient-bg">
        <h1 style='text-align: center; color: white;'>Employee Attrition Predictor</h1>
        <p style='text-align: center; color: white;'>AI-powered analysis to predict employee turnover and help retain your best talent</p>
    </div>
""", unsafe_allow_html=True)

# Layout with two columns
left_col, right_col = st.columns([1, 1])

with left_col:
    with st.form("attrition_form"):
        st.subheader("👤 Employee Details")
        st.markdown("<div class='box-shadow'>", unsafe_allow_html=True)

        satisfaction_level = st.slider("Satisfaction Level (0-100%)", 0, 100, 20)
        st.markdown(f"<p class='slider-label'>Current: {satisfaction_level}%</p>", unsafe_allow_html=True)

        last_evaluation = st.slider("Last Performance Evaluation (0-100%)", 0, 100, 70)
        st.markdown(f"<p class='slider-label'>Current: {last_evaluation}%</p>", unsafe_allow_html=True)

        number_project = st.selectbox("Number of Projects", list(range(1, 11)), index=3)
        department = st.selectbox("Department", [
            'IT', 'Research & Development', 'Accounting', 'Human Resources',
            'Management', 'Marketing', 'Product Management', 'Sales', 'Support', 'Technical'])
        salary = st.selectbox("Salary Level", ['High', 'Medium', 'Low'])
        time_spend_company = st.selectbox("Years at Company", list(range(1, 11)), index=2)
        average_monthly_hours = st.number_input("Monthly Working Hours", min_value=50, max_value=400, value=160)
        work_accident = st.checkbox("Had Work Accident?")
        promotion_last_5years = st.checkbox("Promoted in Last 5 Years?")

        submit = st.form_submit_button("🚀 Predict Attrition Risk")
        st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    if submit:
        input_data = {
            'satisfaction_level': satisfaction_level / 100,
            'last_evaluation': last_evaluation / 100,
            'number_project': number_project,
            'average_monthly_hours': average_monthly_hours,
            'time_spend_company': time_spend_company,
            'Work_accident': 1 if work_accident else 0,
            'promotion_last_5years': 1 if promotion_last_5years else 0
        }

        departments = [
            'IT', 'Research & Development', 'Accounting', 'Human Resources',
            'Management', 'Marketing', 'Product Management', 'Sales', 'Support', 'Technical']
        for dept in departments:
            input_data[dept] = 1 if department == dept else 0

        salary_levels = ['High', 'Medium', 'Low']
        for sal in salary_levels:
            input_data[sal] = 1 if salary == sal else 0

        final_input = pd.DataFrame([input_data])
        final_input = final_input.reindex(columns=features, fill_value=0)


        probs = model.predict_proba(final_input)[0, 1]
        prediction = 1 if probs >= 0.3 else 0

        st.markdown("<div class='box-shadow'>", unsafe_allow_html=True)
        if prediction == 1:
            st.error(f"🚨 High Attrition Risk!\n\n💡 Probability: {probs:.2f}")
            st.markdown("""
                <ul>
                    <li>Schedule a 1:1 meeting to understand concerns.</li>
                    <li>Review compensation & benefits.</li>
                    <li>Evaluate workload and satisfaction.</li>
                    <li>Discuss career development plans.</li>
                </ul>
            """, unsafe_allow_html=True)
        else:
            st.success(f"✅ Low Attrition Risk\n\n📊 Probability: {probs:.2f}")
            st.markdown("""
                <ul>
                    <li>Continue current strategies.</li>
                    <li>Provide appreciation & feedback.</li>
                    <li>Offer growth opportunities.</li>
                    <li>Monitor engagement levels.</li>
                </ul>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Graph
        st.markdown("### 📊 Input Summary")
        input_labels = ["Satisfaction", "Evaluation", "Projects", "Monthly Hours", "Years"]
        input_values = [satisfaction_level, last_evaluation, number_project, average_monthly_hours, time_spend_company]

        fig = go.Figure([go.Bar(x=input_labels, y=input_values, marker_color='orange')])
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #333; color: white; border-radius: 8px; margin-top: 30px;'>
        <p>© 2023 Employee Attrition Predictor. All rights reserved.</p>
        <p>Powered by AI and HR analytics technology.</p>
    </div>
""", unsafe_allow_html=True)
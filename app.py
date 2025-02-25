import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from pyngrok import ngrok


# Load saved models and encoders
lstm_model = load_model('lstm_model.h5')
best_xgb_model = joblib.load('best_xgb_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')
X_train = joblib.load('X_train.pkl')

# Streamlit UI
def main():
    st.title("Employee Attrition Prediction")
    st.write("Predict whether an employee is likely to leave or stay based on input features.")
    
    # Input fields
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    department = st.selectbox("Department", ['Sales', 'Research & Development', 'Human Resources'])
    job_role = st.selectbox("Job Role", ['Sales Executive', 'Research Scientist', 'Laboratory Technician',
                                          'Manufacturing Director', 'Healthcare Representative',
                                          'Manager', 'Sales Representative'])
    job_satisfaction = st.slider("Job Satisfaction", 1, 4, 2)
    marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
    overtime = st.selectbox("OverTime", ['Yes', 'No'])
    percent_salary_hike = st.slider("Percent Salary Hike", 5, 30, 15)
    performance_rating = st.slider("Performance Rating", 1, 4, 3)
    stock_option_level = st.slider("Stock Option Level", 0, 3, 1)
    total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=10)
    work_life_balance = st.slider("Work-Life Balance", 1, 4, 3)
    years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
    years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=20, value=3)
    years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=2)
    
    # Prepare input data
    new_employee_data = {
        'Age': age,
        'Department': department,
        'JobRole': job_role,
        'JobSatisfaction': job_satisfaction,
        'MaritalStatus': marital_status,
        'MonthlyIncome': monthly_income,
        'OverTime': overtime,
        'PercentSalaryHike': percent_salary_hike,
        'PerformanceRating': performance_rating,
        'StockOptionLevel': stock_option_level,
        'TotalWorkingYears': total_working_years,
        'WorkLifeBalance': work_life_balance,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_current_role,
        'YearsSinceLastPromotion': years_since_last_promotion,
    }

    if st.button("Predict Attrition"):
        prediction = predict_attrition(new_employee_data)
        st.subheader("Prediction:")
        st.write(prediction)

# Prediction function
def predict_attrition(new_employee_data):
    new_employee_df = pd.DataFrame([new_employee_data])
    
    # One-hot encode categorical features
    categorical_cols = ['Department', 'JobRole', 'MaritalStatus', 'OverTime']
    for col in categorical_cols:
        onehot_cols = [c for c in X_train.columns if c.startswith(col + '_')]
        for onehot_col in onehot_cols:
            new_employee_df[onehot_col] = 0
        
        value = new_employee_df[col][0]
        matching_col = col + '_' + str(value)
        if matching_col in new_employee_df.columns:
            new_employee_df[matching_col] = 1
    
    new_employee_df.drop(columns=categorical_cols, inplace=True, errors='ignore')
    new_employee_df = new_employee_df.reindex(columns=X_train.columns, fill_value=0)
    
    # LSTM Prediction
    lstm_input = new_employee_df.values.reshape((1, 1, X_train.shape[1]))
    lstm_prediction = lstm_model.predict(lstm_input).flatten()[0]
    
    # XGBoost Prediction
    xgb_prediction = best_xgb_model.predict_proba(new_employee_df)[:, 1][0]
    
    # Hybrid Prediction
    hybrid_prediction = (lstm_prediction + xgb_prediction) / 2
    
    return "Employee is likely to leave." if hybrid_prediction < 0.06 else "Employee is likely to stay."

if __name__ == "__main__":
    main()
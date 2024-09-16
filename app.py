import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.exceptions import InconsistentVersionWarning

# Load the trained model and preprocessor
model = joblib.load(Path('model.joblib'))
churn_preprocess = joblib.load(Path('preprocess.pkl'))

# Create the Streamlit UI with tabs
st.title("Churn Prediction Using Machine Learning")

# Create tabs for "Prediction" and "Feature Explanation"
tabs = st.tabs(["Prediction", "Feature Explanation"])

with tabs[0]:
    st.header("Predict Churn")

    # Input fields for the features
    CreditScore = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
    Geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])
    Gender = st.selectbox("Gender", ['Male', 'Female'])
    Age = st.number_input("Age", min_value=18, max_value=100, value=30)
    Tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)
    Balance = st.number_input("Balance", value=10000.0)
    NumOfProducts = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
    HasCrCard = st.selectbox("Has Credit Card", [0, 1])
    IsActiveMember = st.selectbox("Is Active Member", [0, 1])
    EstimatedSalary = st.number_input("Estimated Salary", value=50000.0)

    # Create a DataFrame from input data
    input_data = pd.DataFrame({
        'CreditScore': [CreditScore],
        'Geography': [Geography],
        'Gender': [Gender],
        'Age': [Age],
        'Tenure': [Tenure],
        'Balance': [Balance],
        'NumOfProducts': [NumOfProducts],
        'HasCrCard': [HasCrCard],
        'IsActiveMember': [IsActiveMember],
        'EstimatedSalary': [EstimatedSalary]
    })

    # Preprocess the input data
    input_data_preprocessed = churn_preprocess.transform(input_data)

    # Predict churn
    if st.button("Predict"):
        prediction = model.predict(input_data_preprocessed)[0]
        if prediction == 1:
            st.warning("This customer is likely to churn.")
        else:
            st.success("This customer is likely to stay.")

with tabs[1]:
    st.header("Feature Explanations")

    st.write("""
    ### Feature Explanations:
    - **CreditScore**: Customer's credit score, ranging from 300 to 850. Higher credit scores indicate better creditworthiness.
    - **Geography**: Customer's location, such as France, Germany, or Spain. Geography can influence customer behavior and churn.
    - **Gender**: Customer's gender (Male or Female). Behavior and churn tendencies might differ between genders.
    - **Age**: Customer's age in years. Younger customers might have higher churn rates, while older customers may be more loyal.
    - **Tenure**: Number of years the customer has been with the company. Higher tenure might indicate stronger loyalty.
    - **Balance**: Account balance. Higher balances could indicate more engagement with the service.
    - **NumOfProducts**: Number of products the customer uses. More products usually mean higher engagement and lower churn.
    - **HasCrCard**: Whether the customer has a credit card with the company (1 = Yes, 0 = No). Having a credit card might reduce churn.
    - **IsActiveMember**: Whether the customer is an active member (1 = Yes, 0 = No). Active members are generally less likely to churn.
    - **EstimatedSalary**: The customer's annual estimated salary. Higher salaries might correlate with higher engagement.
    """)

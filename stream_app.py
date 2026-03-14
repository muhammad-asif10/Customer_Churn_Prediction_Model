import streamlit as st
from PIL import Image
import pandas as pd
import pickle

# Load model
model_file = 'model_C=1.0.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

def main():
    st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")
    
    image = Image.open('images/icone.png')
    st.image(image, width=730)
    st.title("Customer Churn Prediction")
    st.markdown("Predict if a customer is likely to churn and get a risk score.")

    add_selectbox = st.sidebar.selectbox("Prediction Mode", ("Online", "Batch"))
    st.sidebar.info("This app predicts Customer Churn using ML models.")
    st.sidebar.image(Image.open('images/image.png'), width=200)

    if add_selectbox == "Online":
        with st.form("churn_form"):
            st.subheader("Customer Details")
            col1, col2 = st.columns(2)

            with col1:
                gender = st.selectbox('Gender', ['male', 'female'])
                seniorcitizen = st.selectbox('Senior Citizen', [0, 1])
                partner = st.selectbox('Partner', ['yes', 'no'])
                dependents = st.selectbox('Dependents', ['yes', 'no'])
                phoneservice = st.selectbox('Phone Service', ['yes', 'no'])
                multiplelines = st.selectbox('Multiple Lines', ['yes', 'no', 'no_phone_service'])

            with col2:
                internetservice = st.selectbox('Internet Service', ['dsl', 'no', 'fiber_optic'])
                onlinesecurity = st.selectbox('Online Security', ['yes', 'no', 'no_internet_service'])
                onlinebackup = st.selectbox('Online Backup', ['yes', 'no', 'no_internet_service'])
                deviceprotection = st.selectbox('Device Protection', ['yes', 'no', 'no_internet_service'])
                techsupport = st.selectbox('Tech Support', ['yes', 'no', 'no_internet_service'])
                streamingtv = st.selectbox('Streaming TV', ['yes', 'no', 'no_internet_service'])
                streamingmovies = st.selectbox('Streaming Movies', ['yes', 'no', 'no_internet_service'])

            st.subheader("Billing & Contract")
            contract = st.selectbox('Contract', ['month-to-month', 'one_year', 'two_year'])
            paperlessbilling = st.selectbox('Paperless Billing', ['yes', 'no'])
            paymentmethod = st.selectbox('Payment Method', [
                'bank_transfer_(automatic)',
                'credit_card_(automatic)',
                'electronic_check',
                'mailed_check'
            ])
            tenure = st.number_input('Tenure (months)', min_value=0, max_value=240, value=0)
            monthlycharges = st.number_input('Monthly Charges', min_value=0.0, max_value=10000.0, value=0.0, step=1.0)
            totalcharges = tenure * monthlycharges

            submitted = st.form_submit_button("Predict")
            if submitted:
                input_dict = {
                    "gender": gender,
                    "seniorcitizen": seniorcitizen,
                    "partner": partner,
                    "dependents": dependents,
                    "phoneservice": phoneservice,
                    "multiplelines": multiplelines,
                    "internetservice": internetservice,
                    "onlinesecurity": onlinesecurity,
                    "onlinebackup": onlinebackup,
                    "deviceprotection": deviceprotection,
                    "techsupport": techsupport,
                    "streamingtv": streamingtv,
                    "streamingmovies": streamingmovies,
                    "contract": contract,
                    "paperlessbilling": paperlessbilling,
                    "paymentmethod": paymentmethod,
                    "tenure": tenure,
                    "monthlycharges": monthlycharges,
                    "totalcharges": totalcharges
                }
                X = dv.transform([input_dict])
                y_pred = model.predict_proba(X)[0, 1]
                churn = y_pred >= 0.5

                # Risk percent and traffic-light
                risk_percent = int(y_pred * 100)
                if risk_percent < 30:
                    color = "green"
                    icon = "🟢"
                elif risk_percent < 70:
                    color = "yellow"
                    icon = "🟡"
                else:
                    color = "red"
                    icon = "🔴"

                st.markdown(f"**Churn Risk:** {icon} {risk_percent}%")
                # Colored progress bar
                st.markdown(f"""
                <div style="background-color: #e0e0e0; border-radius: 5px; padding: 5px; width: 100%;">
                    <div style="width:{risk_percent}%; background-color:{color}; height:20px; border-radius:5px;"></div>
                </div>
                """, unsafe_allow_html=True)

                st.success(f"Predicted Churn: {churn}")

if __name__ == "__main__":
    main()